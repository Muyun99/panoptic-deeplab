# ------------------------------------------------------------------------------
# Demo code.
# Example command:
# python tools/demo.py --cfg PATH_TO_CONFIG_FILE \
#   --input-files PATH_TO_INPUT_FILES \
#   --output-dir PATH_TO_OUTPUT_DIR
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import argparse
import cv2
import os
import pprint
import logging
import time
import glob
import natsort
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pathlib import Path
import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import save_debug_images
from segmentation.model.post_processing import get_semantic_segmentation, get_panoptic_segmentation
from segmentation.utils import save_annotation, save_instance_annotation, save_panoptic_annotation
import segmentation.data.transforms.transforms as T
from segmentation.utils import AverageMeter
from segmentation.data import build_test_loader_from_cfg

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (path, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            logger.info('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


def read_image(file_name, format=None):
    image = Image.open(file_name)

    # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format == "BGR":
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    if format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)
    return image


def pic2video(filelist, vid_path, size):
    fps = 30
    filelist = natsort.natsorted(filelist, reverse=False)
    print(filelist)
    print(len(filelist))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(vid_path, fourcc, fps, size)

    for item in filelist:
        img = cv2.imread(item)
        video.write(img)

    video.release()


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--input-files',
                        help='input files, could be image, image list or video',
                        required=True,
                        type=str)
    parser.add_argument('--output-dir',
                        help='output directory',
                        required=True,
                        type=str)
    parser.add_argument('--extension',
                        help='file extension if input is image list',
                        default='.png',
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))
    # build model
    model = build_segmentation_model_from_cfg(config)
    logger.info("Model:\n{}".format(model))
    model = model.to(device)

    # build data_loader
    # TODO: still need it for thing_list
    data_loader = build_test_loader_from_cfg(config)

    # load model
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
            logger.info('Evaluating a intermediate checkpoint.')
        model.load_state_dict(model_weights, strict=True)
        logger.info('Test model loaded from {}'.format(model_state_file))
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')

    # dir to save intermediate raw outputs
    raw_out_dir = os.path.join(args.output_dir, 'raw')
    PathManager.mkdirs(raw_out_dir)

    # dir to save semantic outputs
    semantic_out_dir = os.path.join(args.output_dir, 'semantic')
    PathManager.mkdirs(semantic_out_dir)

    # dir to save instance outputs
    instance_out_dir = os.path.join(args.output_dir, 'instance')
    PathManager.mkdirs(instance_out_dir)

    # dir to save panoptic outputs
    panoptic_out_dir = os.path.join(args.output_dir, 'panoptic')
    PathManager.mkdirs(panoptic_out_dir)

    # Test loop
    model.eval()

    # build image demo transform


    net_time = AverageMeter()
    post_time = AverageMeter()

    # dataset
    source = "/home/muyun99/Desktop/MyGithub/cnsoftbei-video/inference/input/video-clip_2-4.mp4"
    imgsz = 1024
    dataset = LoadImages(source, img_size=imgsz)

    try:
        with torch.no_grad():
            for i, (path, image, im0s, vid_cap) in enumerate(dataset):
                (_, raw_h, raw_w) = image.shape
                image = torch.from_numpy(image).to(device)
                image = image.float()
                image /= 255.0  # 0 - 255 to 0.0 - 1.0
                if image.ndimension() == 3:
                    image = image.unsqueeze(0)

                # network
                start_time = time.time()
                out_dict = model(image)
                torch.cuda.synchronize(device)
                net_time.update(time.time() - start_time)

                # post-processing
                start_time = time.time()
                semantic_pred = get_semantic_segmentation(out_dict['semantic'])
                panoptic_pred, center_pred = get_panoptic_segmentation(
                    semantic_pred,
                    out_dict['center'],
                    out_dict['offset'],
                    thing_list=data_loader.dataset.thing_list,
                    label_divisor=data_loader.dataset.label_divisor,
                    stuff_area=config.POST_PROCESSING.STUFF_AREA,
                    void_label=(
                            data_loader.dataset.label_divisor *
                            data_loader.dataset.ignore_label),
                    threshold=config.POST_PROCESSING.CENTER_THRESHOLD,
                    nms_kernel=config.POST_PROCESSING.NMS_KERNEL,
                    top_k=config.POST_PROCESSING.TOP_K_INSTANCE,
                    foreground_mask=None)
                torch.cuda.synchronize(device)
                post_time.update(time.time() - start_time)

                logger.info(
                    'Network Time: {net_time.val:.3f}s ({net_time.avg:.3f}s)\t'
                    'Post-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(
                        net_time=net_time, post_time=post_time)
                )

                # save predictions
                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
                panoptic_pred = panoptic_pred.squeeze(0).cpu().numpy()

                # crop predictions
                semantic_pred = semantic_pred[:raw_h, :raw_w]
                panoptic_pred = panoptic_pred[:raw_h, :raw_w]

                # Raw outputs
                save_debug_images(
                    dataset=data_loader.dataset,
                    batch_images=image,
                    batch_targets={},
                    batch_outputs=out_dict,
                    out_dir=raw_out_dir,
                    iteration=i,
                    target_keys=[],
                    output_keys=['semantic', 'center', 'offset'],
                    is_train=False,
                )

                save_annotation(semantic_pred, semantic_out_dir, 'semantic_pred_%d' % i,
                                add_colormap=True, colormap=data_loader.dataset.create_label_colormap())
                pan_to_sem = panoptic_pred // data_loader.dataset.label_divisor
                save_annotation(pan_to_sem, semantic_out_dir, 'pan_to_sem_pred_%d' % i,
                                add_colormap=True, colormap=data_loader.dataset.create_label_colormap())
                ins_id = panoptic_pred % data_loader.dataset.label_divisor
                pan_to_ins = panoptic_pred.copy()
                pan_to_ins[ins_id == 0] = 0
                save_instance_annotation(pan_to_ins, instance_out_dir, 'pan_to_ins_pred_%d' % i)
                save_panoptic_annotation(panoptic_pred, panoptic_out_dir, 'panoptic_pred_%d' % i,
                                         label_divisor=data_loader.dataset.label_divisor,
                                         colormap=data_loader.dataset.create_label_colormap())
    except Exception:
        logger.exception("Exception during demo:")
        raise
    finally:
        logger.info("Demo finished.")

    vid_path = args.output_dir
    pic2video(filelist=glob.glob(semantic_out_dir + "pan_to_sem_pred_*"),
              vid_path=os.path.join(vid_path, "semantic.mp4"),
              size=(1024, 576))
    pic2video(filelist=glob.glob(instance_out_dir + "pan_to_ins_pred_*"),
              vid_path=os.path.join(vid_path, "instance.mp4"),
              size=(1024, 576))
    pic2video(filelist=glob.glob(panoptic_out_dir + "panoptic_pred_*"),
              vid_path=os.path.join(vid_path, "panoptic.mp4"),
              size=(1024, 576))
    logger.info("Video saved")


if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger('demo')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=args.output_dir, name='demo')

    logger.info(pprint.pformat(args))
    logger.info(config)
    main()