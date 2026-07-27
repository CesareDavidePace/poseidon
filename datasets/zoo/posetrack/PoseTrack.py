#!/usr/bin/python
# -*- coding:utf8 -*-
from random import choice

import numpy as np
import os.path as osp
import torch
import copy
import random
import cv2
from pycocotools.coco import COCO
import logging
from collections import OrderedDict
from tabulate import tabulate
from termcolor import colored

from .posetrack_utils.posetrack_utils import video2filenames
from .posetrack_utils.poseval.py import evaluate_simple
from utils.utils_json import read_json_from_file, write_json_to_file
from utils.utils_bbox import box2cs
from utils.utils_image import read_image, read_image_pil, read_images_parallel
from utils.utils_folder import create_folder
from utils.utils_registry import DATASET_REGISTRY
from datasets.process import get_affine_transform, fliplr_joints, exec_affine_transform, generate_heatmaps, half_body_transform, \
    convert_data_to_annorect_struct

from datasets.transforms import build_transforms
from datasets.zoo.base import VideoDataset

from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE

from .target_generator import HeatmapGenerator
from concurrent.futures import ThreadPoolExecutor

import cProfile
import pstats
from pstats import SortKey
from PIL import Image
import numpy as np
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import shutil 

import torch.nn.functional as F
from torchvision.transforms import functional as TF

def check_file_and_read(img_file):
    if not osp.exists(img_file):
        error_msg = f"Can not find image: {img_file}"
        logger = logging.getLogger(__name__)
        logger.error(error_msg)
        raise Exception(error_msg)
    return read_image_pil(img_file)


class PoseTrack(VideoDataset):
    """
        PoseTrack
    """

    def __init__(self, cfg, phase, **kwargs):
        super(PoseTrack, self).__init__(cfg, phase, **kwargs)

        # print Dataset loading in green
        print("\033[92m" + "Dataset: " + "\033[0m", "PoseTrack")

        self.train = True if phase == TRAIN_PHASE else False
        self.flip_pairs = [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

        self.joints_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5], dtype=np.float32).reshape((self.num_joints, 1)) 
        self.joints_with_center_weight = np.array([1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5, 1.5],
                                    dtype=np.float32).reshape((self.num_joints + 1, 1))

        self.dataset_name = cfg.DATASET.NAME
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)
        self.is_posetrack18 = cfg.DATASET.IS_POSETRACK18
        self.transform = build_transforms(cfg, phase)

        self.distance_whole_otherwise_segment = cfg.DISTANCE_WHOLE_OTHERWISE_SEGMENT
        self.distance = cfg.DISTANCE
        self.previous_distance = cfg.PREVIOUS_DISTANCE
        self.next_distance = cfg.NEXT_DISTANCE
        self.window_size = cfg.WINDOWS_SIZE

        self.random_aux_frame = cfg.DATASET.RANDOM_AUX_FRAME

        self.bbox_enlarge_factor = cfg.DATASET.BBOX_ENLARGE_FACTOR
        self.sigma = cfg.MODEL.SIGMA

        self.img_dir = cfg.DATASET.IMG_DIR
        self.json_dir = cfg.DATASET.JSON_DIR
        self.test_on_train = cfg.DATASET.TEST_ON_TRAIN
        self.json_file = cfg.DATASET.JSON_FILE

        self.image_format = cfg.IMAGE_FORMAT

        if self.phase != TRAIN_PHASE:
            self.img_dir = cfg.DATASET.TEST_IMG_DIR
            temp_subCfgNode = cfg.VAL if self.phase == VAL_PHASE else cfg.TEST
            self.nms_thre = temp_subCfgNode.NMS_THRE
            self.image_thre = temp_subCfgNode.IMAGE_THRE
            self.soft_nms = temp_subCfgNode.SOFT_NMS
            self.oks_thre = temp_subCfgNode.OKS_THRE
            self.in_vis_thre = temp_subCfgNode.IN_VIS_THRE
            self.bbox_file = temp_subCfgNode.COCO_BBOX_FILE
            self.use_gt_bbox = temp_subCfgNode.USE_GT_BBOX
            self.annotation_dir = temp_subCfgNode.ANNOT_DIR

        self.coco = COCO(osp.join(self.json_dir, 'posetrack_train.json' if self.is_train else 'posetrack_val.json'))
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])
        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)

        self.data = self._list_data()

        self.model_input_type = cfg.DATASET.INPUT_TYPE

        self.show_data_parameters()
        self.show_samples()

        self.motion_augmentation = cfg.TRAIN.MOTION_AUGMENTATION

        print("motion_augmentation: ", self.motion_augmentation)

        #CID
        self.output_size = cfg.MODEL.HEATMAP_SIZE[0]
        self.heatmap_generator = HeatmapGenerator(cfg.MODEL.HEATMAP_SIZE[0])

        print("\n\n")

        # check if is train or val
        if self.train:
            print("\033[92m" + "Train dataset loaded successfully." + "\033[0m")
        else:
            print("\033[92m" + "Val dataset loaded successfully." + "\033[0m")


    def __getitem__(self, item_index):

        #data_item = copy.deepcopy(self.data[item_index])
        data_item = self.data[item_index]

        if self.model_input_type == 'single_frame':
            return self._get_single_frame(data_item)
        elif self.model_input_type == 'spatiotemporal_window':
            if self.motion_augmentation:
                return self._get_spatiotemporal_window_multi_aug(data_item)
            else:
                return self._get_spatiotemporal_window_multi(data_item)

    def _get_spatiotemporal_window(self, data_item):

        profiler = cProfile.Profile()
        profiler.enable()

    
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        #data_numpy = read_image(image_file_path)

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        is_posetrack18 = zero_fill == 6

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        prev_idx = max((current_idx - 1), 0)
        next_idx = min(current_idx + 1, num_frames-1)

        prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.jpg')
        next_image_file = osp.join(osp.dirname(image_file_path), str(next_idx).zfill(zero_fill) + '.jpg')

        image_files = [prev_image_file, image_file_path, next_image_file]

        #data_numpy_prev = read_image(prev_image_file)
        #data_numpy_next = read_image(next_image_file)

        data_numpy_prev, data_numpy, data_numpy_next = read_images_parallel(image_files)

        if data_numpy is None or data_numpy_prev is None or data_numpy_next is None:
            error_files = [image_file_path, prev_image_file, next_image_file]
            for file, data in zip(error_files, [data_numpy, data_numpy_prev, data_numpy_next]):
                if data is None:
                    self.logger.error(f"Fail to read {file}")
                    raise ValueError(f"Fail to read {file}")

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if self.train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_prev = data_numpy_prev[:, ::-1, :]
                data_numpy_next = data_numpy_next[:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev = cv2.warpAffine(data_numpy_prev, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_next = cv2.warpAffine(data_numpy_next, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev = self.transform(input_prev)
            input_next = self.transform(input_next)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)


        meta = {
            'image': image_file_path,
            #'target': target,
            #'target_weight': target_weight,
            'prev_sup_image': prev_image_file,
            'next_sup_image': next_image_file,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
        }

        x = torch.stack([input_prev, input_x, input_next], dim=1)

        #return input_prev, input_x, input_next, meta, target_heatmaps, target_heatmaps_weight

        return x, meta, target_heatmaps, target_heatmaps_weight


    def _get_spatiotemporal_window_old(self, data_item):
        
        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']
        data_numpy = read_image(image_file_path)

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))

        is_posetrack18 = zero_fill == 6

        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        prev_idx = max((current_idx - 1), 0)
        next_idx = min(current_idx + 1, num_frames-1)

        prev_image_file = osp.join(osp.dirname(image_file_path), str(prev_idx).zfill(zero_fill) + '.jpg')
        next_image_file = osp.join(osp.dirname(image_file_path), str(next_idx).zfill(zero_fill) + '.jpg')

        data_numpy_prev = read_image(prev_image_file)
        data_numpy_next = read_image(next_image_file)

        if data_numpy is None or data_numpy_prev is None or data_numpy_next is None:
            error_files = [image_file_path, prev_image_file, next_image_file]
            for file, data in zip(error_files, [data_numpy, data_numpy_prev, data_numpy_next]):
                if data is None:
                    self.logger.error(f"Fail to read {file}")
                    raise ValueError(f"Fail to read {file}")

        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']

        center = data_item["center"]
        scale = data_item["scale"]

        score = data_item['score'] if 'score' in data_item else 1
        r = 0

        if False: #if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                data_numpy_prev = data_numpy_prev[:, ::-1, :]
                data_numpy_next = data_numpy_next[:, ::-1, :]

                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        # calculate transform matrix
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_x = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_prev = cv2.warpAffine(data_numpy_prev, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        input_next = cv2.warpAffine(data_numpy_next, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        if self.transform:
            input_x = self.transform(input_x)
            input_prev = self.transform(input_prev)
            input_next = self.transform(input_next)

        # joint transform like image
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:  # joints_vis   num_joints  x 3 (x_vis,y_vis)
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        # H W
        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]
        # target_heatmaps, target_heatmaps_weight = self._generate_target(joints, joints_vis, self.heatmap_size, self.num_joints)

        target_heatmaps, target_heatmaps_weight = generate_heatmaps(joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
                                                                    self.num_joints,
                                                                    use_different_joints_weight=self.use_different_joints_weight,
                                                                    joints_weight=self.joints_weight)
        target_heatmaps = torch.from_numpy(target_heatmaps)  # H W
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)


        meta = {
            'image': image_file_path,
            #'target': target,
            #'target_weight': target_weight,
            'prev_sup_image': prev_image_file,
            'next_sup_image': next_image_file,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
        }

        x = torch.stack([input_prev, input_x, input_next], dim=1)

        return x, meta, target_heatmaps, target_heatmaps_weight
    
    def _get_spatiotemporal_window_multi_aug(self, data_item):

        #print("Using motion augmentation")

        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']

        zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))
        current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))

        # Calculate indices for the temporal window
        half_window = self.window_size // 2

        if self.train:
            start_idx = current_idx - half_window
            end_idx = min(current_idx + half_window, 10000)

            # Generate list of image files for the temporal window
            image_files = [
                osp.join(osp.dirname(image_file_path), f"{str(i).zfill(zero_fill)}.jpg")
                for i in range(start_idx, end_idx + 1)
            ]
        else:
            start_idx = current_idx - half_window
            end_idx = current_idx + half_window

            image_files = []
            for i in range(start_idx, end_idx + 1):
                if i < 0:
                    image_files.append(osp.join(osp.dirname(image_file_path), f"{str(0).zfill(zero_fill)}.jpg"))
                elif i >= num_frames:
                    image_files.append(osp.join(osp.dirname(image_file_path), f"{str(num_frames-1).zfill(zero_fill)}.jpg"))
                else:
                    image_files.append(osp.join(osp.dirname(image_file_path), f"{str(i).zfill(zero_fill)}.jpg"))


        # Read images in parallel
        data_numpy_list = read_images_parallel(image_files)

        # Check if any image failed to load
        if any(img is None for img in data_numpy_list):
            error_files = [file for file, img in zip(image_files, data_numpy_list) if img is None]
            for file in error_files:
                self.logger.error(f"Failed to read {file}")
            raise ValueError(f"Failed to read one or more images")

        # Get the central frame
        central_frame = data_numpy_list[len(data_numpy_list) // 2]

        if self.train:
            # Temporal Augmentation: Frame Dropping with Noise
            temporal_drop_prob = 0.3  # Probability of dropping frames
            max_frames_to_drop = 2
            if random.random() < temporal_drop_prob and len(data_numpy_list) > 2:
                #print("Dropping frames")
                num_frames_to_drop = random.randint(1, min(max_frames_to_drop, len(data_numpy_list) - 1))
                frames_to_drop = random.sample(range(len(data_numpy_list)), num_frames_to_drop)
                #print(f"Dropping frames: {frames_to_drop}")
                for idx in frames_to_drop:
                    noise = np.random.normal(loc=0.0, scale=20, size=data_numpy_list[idx].shape)
                    noisy_frame = data_numpy_list[idx] + noise
                    noisy_frame = np.clip(noisy_frame, 0, 255)  # Ensure pixel values are valid
                    data_numpy_list[idx] = noisy_frame.astype(data_numpy_list[idx].dtype)
        
        # if self.train:
        #     # Temporal Augmentation: Temporal Shuffling
        #     temporal_shuffle_prob = 0.0  # Probability of shuffling frames
        #     if random.random() < temporal_shuffle_prob:
        #         print("Shuffling frames")
        #         frame_order = list(range(len(data_numpy_list)))
        #         random.shuffle(frame_order)
        #         data_numpy_list = [data_numpy_list[i] for i in frame_order]
        #         # Update central frame after shuffling
        #         central_frame_idx = frame_order.index(len(data_numpy_list) // 2)
        #         central_frame = data_numpy_list[central_frame_idx]

        if self.train:
            # Temporal Augmentation: Time Reversal
            time_reversal_prob = 0.3  # Probability of reversing frames
            if random.random() < time_reversal_prob:
                data_numpy_list = data_numpy_list[::-1]
                # Update central frame after reversal
                central_frame_idx = len(data_numpy_list) // 2
                central_frame = data_numpy_list[central_frame_idx]

        if self.train:
            # Temporal Augmentation: Frame-Level CutMix
            cutmix_prob = 0.1  # Probability of applying CutMix
            if random.random() < cutmix_prob:
                #print("Applying CutMix between frames")
                frame1_idx, frame2_idx = random.sample(range(len(data_numpy_list)), 2)  # Select two random frames
                lam = np.random.beta(1.0, 1.0)  # Lambda for mixing
                h, w, _ = data_numpy_list[frame1_idx].shape
                cut_h, cut_w = int(lam * h), int(lam * w)
                data_numpy_list[frame1_idx][:cut_h, :cut_w] = data_numpy_list[frame2_idx][:cut_h, :cut_w]  # Mix parts of frames

        # if self.train:
        #     # Temporal Augmentation: Temporal Cropping
        #     temporal_crop_prob = 1  # Probability of cropping frames
        #     min_seq_length = max(3, int(len(data_numpy_list) * 0.8))  # Keep at least 80% of frames, minimum 3 frames
        #     if random.random() < temporal_crop_prob and len(data_numpy_list) > min_seq_length:
        #         print("Cropping frames")
        #         start_frame = random.randint(0, len(data_numpy_list) - min_seq_length)
        #         end_frame = start_frame + min_seq_length
        #         data_numpy_list = data_numpy_list[start_frame:end_frame]
        #         # Update central frame after cropping
        #         central_frame_idx = len(data_numpy_list) // 2
        #         central_frame = data_numpy_list[central_frame_idx]

        # if self.train:
        #     # Temporal Augmentation: Speed Variation
        #     speed_variation_prob = 0  # Probability of changing speed
        #     if random.random() < speed_variation_prob:
        #         print("Changing speed")
        #         speed_factor = random.choice([0.5, 1.5])  # Slow down (duplicate frames) or speed up (skip frames)
        #         indices = np.linspace(0, len(data_numpy_list) - 1, int(len(data_numpy_list) * speed_factor))
        #         indices = np.clip(indices.astype(int), 0, len(data_numpy_list) - 1)
        #         data_numpy_list = [data_numpy_list[i] for i in indices]
        #         # Update central frame after speed variation
        #         central_frame_idx = len(data_numpy_list) // 2
        #         central_frame = data_numpy_list[central_frame_idx]

        # Rest of the processing (joints, center, scale, etc.)
        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']
        center = data_item["center"]
        scale = data_item["scale"]
        score = data_item.get('score', 1)
        r = 0

        if self.train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy_list = [img[:, ::-1, :] for img in data_numpy_list]

                joints, joints_vis = fliplr_joints(joints, joints_vis, central_frame.shape[1], self.flip_pairs)
                center[0] = central_frame.shape[1] - center[0] - 1

        # Apply affine transform to all images in the window
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_images = [
            cv2.warpAffine(img, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
            for img in data_numpy_list
        ]

        if self.transform:
            input_images = [self.transform(img) for img in input_images]

        # Joint transform and visibility check
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        # Generate target heatmaps
        target_heatmaps, target_heatmaps_weight = generate_heatmaps(
            joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
            self.num_joints, use_different_joints_weight=self.use_different_joints_weight,
            joints_weight=self.joints_weight
        )
        
        target_heatmaps = torch.from_numpy(target_heatmaps)
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        # Prepare metadata
        meta = {
            'image': image_file_path,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
        }

        # Stack input images
        x = torch.stack(input_images, dim=0)

        return x, meta, target_heatmaps, target_heatmaps_weight
        
    def _get_spatiotemporal_window_multi(self, data_item):

        filename = data_item['filename']
        img_num = data_item['imgnum']
        image_file_path = data_item['image']
        num_frames = data_item['nframes']

        if self.image_format  == 'jpg':
            zero_fill = len(osp.basename(image_file_path).replace('.jpg', ''))
            current_idx = int(osp.basename(image_file_path).replace('.jpg', ''))
        elif self.image_format  == 'png':
            zero_fill = len(osp.basename(image_file_path).replace('.png', ''))
            current_idx = int(osp.basename(image_file_path).replace('.png', ''))

        # Calculate indices for the temporal window
        half_window = self.window_size // 2

        if self.train:
            start_idx = current_idx - half_window
            end_idx = min(current_idx + half_window, 10000)

            # Generate list of image files for the temporal window
            image_files = [
                osp.join(osp.dirname(image_file_path), f"{str(i).zfill(zero_fill)}.{self.image_format}")
                for i in range(start_idx, end_idx + 1)
            ]

            if self.dataset_name == 'jhmdb':
                image_files = []
                for i in range(start_idx, end_idx + 1):
                    if i < 1:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(1).zfill(zero_fill)}.{self.image_format}"))
                    elif i >= num_frames:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(num_frames-1).zfill(zero_fill)}.{self.image_format}"))
                    else:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(i).zfill(zero_fill)}.{self.image_format}"))

        else:
            start_idx = current_idx - half_window
            end_idx = current_idx + half_window

            if self.dataset_name == 'jhmdb':
                image_files = []
                for i in range(start_idx, end_idx + 1):
                    if i < 1:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(1).zfill(zero_fill)}.{self.image_format}"))
                    elif i >= num_frames:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(num_frames-1).zfill(zero_fill)}.{self.image_format}"))
                    else:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(i).zfill(zero_fill)}.{self.image_format}"))
            else:

                image_files = []
                for i in range(start_idx, end_idx + 1):
                    if i < 0:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(0).zfill(zero_fill)}.{self.image_format}"))
                    elif i >= num_frames:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(num_frames-1).zfill(zero_fill)}.{self.image_format}"))
                    else:
                        image_files.append(osp.join(osp.dirname(image_file_path), f"{str(i).zfill(zero_fill)}.{self.image_format}"))

        if self.image_format == 'jpg':
            # Read images in parallel
            data_numpy_list = read_images_parallel(image_files)

        elif self.image_format == 'png':
            data_numpy_list = read_images_parallel(image_files, format='png')

        # Check if any image failed to load
        if any(img is None for img in data_numpy_list):
            error_files = [file for file, img in zip(image_files, data_numpy_list) if img is None]
            for file in error_files:
                self.logger.error(f"Failed to read {file}")
            raise ValueError(f"Failed to read one or more images")

        # Get the central frame
        central_frame = data_numpy_list[len(data_numpy_list) // 2]

        # Rest of the processing (joints, center, scale, etc.)
        joints = data_item['joints_3d']
        joints_vis = data_item['joints_3d_vis']
        center = data_item["center"]
        scale = data_item["scale"]
        score = data_item.get('score', 1)
        r = 0

        if self.train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = half_body_transform(joints, joints_vis, self.num_joints, self.upper_body_ids, self.aspect_ratio,
                                                               self.pixel_std)
                center, scale = c_half_body, s_half_body

            scale_factor = self.scale_factor
            rotation_factor = self.rotation_factor
            # scale = scale * np.random.uniform(1 - scale_factor[0], 1 + scale_factor[1])
            if isinstance(scale_factor, list) or isinstance(scale_factor, tuple):
                scale_factor = scale_factor[0]
            scale = scale * np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
            r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy_list = [img[:, ::-1, :] for img in data_numpy_list]

                joints, joints_vis = fliplr_joints(joints, joints_vis, central_frame.shape[1], self.flip_pairs)
                center[0] = central_frame.shape[1] - center[0] - 1

        # Apply affine transform to all images in the window
        trans = get_affine_transform(center, scale, r, self.image_size)
        input_images = [
            cv2.warpAffine(img, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
            for img in data_numpy_list
        ]

        if self.transform:
            input_images = [self.transform(img) for img in input_images]

        # Joint transform and visibility check
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = exec_affine_transform(joints[i, 0:2], trans)

        for index, joint in enumerate(joints):
            x, y, _ = joint
            if x < 0 or y < 0 or x > self.image_size[0] or y > self.image_size[1]:
                joints_vis[index] = [0, 0, 0]

        # Generate target heatmaps
        target_heatmaps, target_heatmaps_weight = generate_heatmaps(
            joints, joints_vis, self.sigma, self.image_size, self.heatmap_size,
            self.num_joints, use_different_joints_weight=self.use_different_joints_weight,
            joints_weight=self.joints_weight
        )
        
        target_heatmaps = torch.from_numpy(target_heatmaps)
        target_heatmaps_weight = torch.from_numpy(target_heatmaps_weight)

        # Prepare metadata
        meta = {
            'image': image_file_path,
            'filename': filename,
            'imgnum': img_num,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': center,
            'scale': scale,
            'rotation': r,
            'score': score,
        }

        # Stack input images
        x = torch.stack(input_images, dim=0)

        return x, meta, target_heatmaps, target_heatmaps_weight

    def _generate_target(self, joints_3d, joints_3d_visible):
        """Generate the target regression vector.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray([num_joints, 3])
            joints_3d_visible: np.ndarray([num_joints, 3])

        Returns:
             target, target_weight(1: visible, 0: invisible)
        """
        image_size = self.image_size
        joint_weights = self.joints_weight
        use_different_joint_weights = self.use_different_joints_weight
        assert use_different_joint_weights is True

        mask = (joints_3d[:, 0] >= 0) * (
            joints_3d[:, 0] <= image_size[0] - 1) * (joints_3d[:, 1] >= 0) * (
                joints_3d[:, 1] <= image_size[1] - 1)

        target = joints_3d[:, :2] / image_size

        target = target.astype(np.float32)
        target_weight = joints_3d_visible[:, :2] * mask[:, None]

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def _get_single_frame(self, data_item):
        raise NotImplementedError

    def _list_data(self):
        if self.is_train or self.use_gt_bbox:
            # use bbox from annotation
            data = self._load_coco_keypoints_annotations()
        else:
            # use bbox from detection
            data = self._load_detection_results()
        return data

    def _load_coco_keypoints_annotations(self):
        """
            coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
            iscrowd:
                crowd instances are handled by marking their overlaps with all categories to -1
                and later excluded in training
            bbox:
                [x1, y1, w, h]
        """
        gt_db = []
        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            width = im_ann['width']
            height = im_ann['height']

            file_name = im_ann['file_name']

            nframes = int(im_ann['nframes'])
            frame_id = int(im_ann['frame_id'])

            annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
            objs = self.coco.loadAnns(annIds)

            # sanitize bboxes
            valid_objs = []
            for obj in objs:
                x, y, w, h = obj['bbox']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                    obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
                    valid_objs.append(obj)
            objs = valid_objs

            rec = []
            for obj in objs:
                cls = self._coco_ind_to_class_ind[obj['category_id']]
                if cls != 1:
                    continue

                # ignore objs without keypoints annotation
                if max(obj['keypoints']) == 0:
                    continue

                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
                joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float32)
                for ipt in range(self.num_joints):
                    joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                    joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                    joints_3d[ipt, 2] = 0
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis > 1:
                        t_vis = 1
                    joints_3d_vis[ipt, 0] = t_vis
                    joints_3d_vis[ipt, 1] = t_vis
                    joints_3d_vis[ipt, 2] = 0

                center, scale = box2cs(obj['clean_bbox'][:4], self.aspect_ratio, self.bbox_enlarge_factor)
                rec.append({
                    'image': osp.join(self.img_dir, file_name),
                    'center': center,
                    'scale': scale,
                    'box': obj['clean_bbox'][:4],
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'nframes': nframes,
                    'frame_id': frame_id,
                })
            gt_db.extend(rec)
        return gt_db

    def _load_detection_results(self):
        logger = logging.getLogger(__name__)
        logger.info("=> Load bbox file from {}".format(self.bbox_file))
        all_boxes = read_json_from_file(self.bbox_file)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        kpt_data = []
        num_boxes = 0

        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = det_res['image_name']
            box = det_res['bbox']  # xywh
            score = det_res['score']
            nframes = det_res['nframes']
            frame_id = det_res['frame_id']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = box2cs(box, self.aspect_ratio, self.bbox_enlarge_factor)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float32)
            kpt_data.append({
                'image': osp.join(self.img_dir, img_name),
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'nframes': nframes,
                'frame_id': frame_id,
            })

        table_header = ["Total boxes", "Filter threshold", "Remaining boxes"]
        table_data = [[len(all_boxes), self.image_thre, num_boxes]]
        table = tabulate(table_data, tablefmt="pipe", headers=table_header, numalign="left")
        logger.info(f"=> Boxes Info Table: \n" + colored(table, "magenta"))

        return kpt_data

    def evaluate(self, cfg, preds, output_dir, boxes, img_path, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("=> Start evaluate")
        if self.phase == 'validate':
            output_dir = osp.join(output_dir, 'val_set_json_results_jhmdb')
        else:
            output_dir = osp.join(output_dir, 'test_set_json_results')

        create_folder(output_dir)

        ### processing our preds
        video_map = {}
        vid2frame_map = {}
        vid2name_map = {}

        all_preds = []
        all_boxes = []
        all_tracks = []
        cc = 0

        # print(img_path)
        for key in img_path:
            temp = key.split('/')

            # video_name = osp.dirname(key)
            video_name = temp[len(temp) - 3] + '/' + temp[len(temp) - 2]
            img_sfx = temp[len(temp) - 3] + '/' + temp[len(temp) - 2] + '/' + temp[len(temp) - 1]

            prev_nm = temp[len(temp) - 1]
            frame_num = int(prev_nm.replace(str('.' + self.image_format), ''))
            if not video_name in video_map:
                video_map[video_name] = [cc]
                vid2frame_map[video_name] = [frame_num]
                vid2name_map[video_name] = [img_sfx]
            else:
                video_map[video_name].append(cc)
                vid2frame_map[video_name].append(frame_num)
                vid2name_map[video_name].append(img_sfx)

            idx_list = img_path[key]
            pose_list = []
            box_list = []
            for idx in idx_list:
                temp = np.zeros((4, 17))
                temp[0, :] = preds[idx, :, 0]
                temp[1, :] = preds[idx, :, 1]
                temp[2, :] = preds[idx, :, 2]
                temp[3, :] = preds[idx, :, 2]
                pose_list.append(temp)

                temp = np.zeros((1, 6))
                temp[0, :] = boxes[idx, :]
                box_list.append(temp)

            all_preds.append(pose_list)
            all_boxes.append(box_list)
            cc += 1

        annot_dir = self.annotation_dir
        is_posetrack18 = self.is_posetrack18

        out_data = {}
        out_filenames, L = video2filenames(annot_dir)

        for vid in video_map:
            idx_list = video_map[vid]
            
            c = 0
            used_frame_list = []

            if self.dataset_name == 'posetrack':
                cur_length = L['images/' + vid]
            elif self.dataset_name == 'jhmdb':
                cur_length = L['Rename_Images/' + vid]

            temp_kps_map = {}
            temp_track_kps_map = {}
            temp_box_map = {}

            for idx in idx_list:
                frame_num = vid2frame_map[vid][c]
                img_sfx = vid2name_map[vid][c]
                c += 1

                used_frame_list.append(frame_num)

                kps = all_preds[idx]
                temp_kps_map[frame_num] = (img_sfx, kps)

                bb = all_boxes[idx]
                temp_box_map[frame_num] = bb
            #### including empty frames
            nnz_counter = 0
            next_track_id = 0

            if not is_posetrack18:
                sid = 1
                fid = cur_length + 1
            else:
                sid = 0
                fid = cur_length
            # start id ~ finish id
            for current_frame_id in range(sid, fid):
                frame_num = current_frame_id
                if not current_frame_id in used_frame_list:
                    temp_sfx = vid2name_map[vid][0]
                    arr = temp_sfx.split('/')
                    if not is_posetrack18:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(8) + '.jpg'
                    else:
                        img_sfx = arr[0] + '/' + arr[1] + '/' + str(frame_num).zfill(6) + '.jpg'
                    kps = []
                    tracks = []
                    bboxs = []

                else:

                    img_sfx = temp_kps_map[frame_num][0]
                    kps = temp_kps_map[frame_num][1]
                    bboxs = temp_box_map[frame_num]
                    tracks = [track_id for track_id in range(len(kps))]
                    # tracks = [1] * len(kps)

                ### creating a data element
                data_el = {
                    'image': {'name': img_sfx},
                    'imgnum': [frame_num],
                    'annorect': convert_data_to_annorect_struct(kps, tracks, bboxs, self.dataset_name),
                }
                if vid in out_data:
                    out_data[vid].append(data_el)
                else:
                    out_data[vid] = [data_el]

        logger.info("=> saving files for evaluation")
        #### saving files for evaluation
        for vname in out_data:
            vdata = out_data[vname]

            if self.dataset_name == "posetrack":
                outfpath = osp.join(output_dir, out_filenames[osp.join('images', vname)])
            elif self.dataset_name == "jhmdb":
                outfpath = osp.join(output_dir, out_filenames[osp.join('Rename_Images', vname)])

            write_json_to_file({'annolist': vdata}, outfpath)

        # run evaluation
        # AP = self._run_eval(annot_dir, output_dir)[0]
        AP = evaluate_simple.evaluate(annot_dir, output_dir, eval_track=False, dataset_name=self.dataset_name)[0]
        
        # Calculate and save per-image mAP for validation set
        if self.phase == 'validate':
            per_image_map = self._calculate_per_image_map(annot_dir, output_dir, img_path)
            per_image_map_file = osp.join(output_dir, 'per_image_map.json')
            write_json_to_file(per_image_map, per_image_map_file)
            logger.info(f"=> Per-image mAP saved to: {per_image_map_file}")
        
        if self.dataset_name == "posetrack":
            name_value = [
                ('Head', AP[0]),
                ('Shoulder', AP[1]),
                ('Elbow', AP[2]),
                ('Wrist', AP[3]),
                ('Hip', AP[4]),
                ('Knee', AP[5]),
                ('Ankle', AP[6]),
                ('Mean', AP[7])
            ]

            name_value = OrderedDict(name_value)

            return name_value, name_value['Mean']

        else:
            return AP, AP["Mean"]

    def _calculate_per_image_map(self, annot_dir, output_dir, img_path):
        """
        Calculate mAP for each individual image in the validation set.
        
        Args:
            annot_dir: Directory containing ground truth annotations
            output_dir: Directory containing prediction results
            img_path: Dictionary mapping image paths to prediction indices
            
        Returns:
            dict: Dictionary containing per-image mAP results
        """
        logger = logging.getLogger(__name__)
        logger.info("=> Calculating per-image mAP")
        
        from .posetrack_utils.poseval.py.evaluateAP import evaluateAP
        from .posetrack_utils.poseval.py.eval_helpers import load_data_dir
        
        # Load all ground truth and prediction data
        if self.dataset_name == "posetrack":
            gtFramesAll, prFramesAll = load_data_dir(['', annot_dir, output_dir])
        else:
            from .posetrack_utils.poseval.py.evaluate_simple import load_data_dir_jhmdb
            gtFramesAll, prFramesAll = load_data_dir_jhmdb(['', annot_dir, output_dir])
        
        # Create dictionaries for fast lookup
        gt_frames_dict = {}
        for frame in gtFramesAll:
            image_name = frame['image'][0]['name'] if isinstance(frame['image'], list) else frame['image']['name']
            if image_name not in gt_frames_dict:
                gt_frames_dict[image_name] = []
            gt_frames_dict[image_name].append(frame)
        
        pr_frames_dict = {}
        for frame in prFramesAll:
            image_name = frame['image'][0]['name'] if isinstance(frame['image'], list) else frame['image']['name']
            if image_name not in pr_frames_dict:
                pr_frames_dict[image_name] = []
            pr_frames_dict[image_name].append(frame)
        
        per_image_results = {}
        
        # Calculate mAP for each image
        for img_key in img_path:
            # Extract image filename from path
            img_filename = '/'.join(img_key.split('/')[-3:])  # Get the relative path format
            
            # Find corresponding ground truth and prediction frames
            gt_frames_img = gt_frames_dict.get(img_filename, [])
            pr_frames_img = pr_frames_dict.get(img_filename, [])
            
            if len(gt_frames_img) == 0:
                logger.warning(f"No ground truth found for image: {img_filename}")
                continue
                
            if len(pr_frames_img) == 0:
                logger.warning(f"No predictions found for image: {img_filename}")
                per_image_results[img_key] = {
                    'image_path': img_key,
                    'mAP': 0.0,
                    'joint_APs': [0.0] * self.num_joints,
                    'has_predictions': False
                }
                continue
            
            # Calculate AP for this specific image
            try:
                if self.dataset_name == "posetrack":
                    apImg, _, _ = evaluateAP(gt_frames_img, pr_frames_img)
                    joint_aps = [float(ap) if not np.isnan(ap) else 0.0 for ap in apImg[:, 0]]
                    mean_ap = np.nanmean(apImg[:, 0]) if len(apImg) > 0 else 0.0
                    mean_ap = float(mean_ap) if not np.isnan(mean_ap) else 0.0
                else:
                    # For JHMDB, use PCK metric instead
                    keypoint_names = [
                        "neck", "belly", "head", "left_ear", "right_ear",
                        "right_shoulder", "left_shoulder", "right_hip", "left_hip",
                        "right_elbow", "left_elbow", "right_knee", "left_knee",
                        "right_wrist", "left_wrist", "right_ankle", "left_ankle"
                    ]
                    from .posetrack_utils.poseval.py.evaluate_simple import calculate_pck
                    pck_results = calculate_pck(gt_frames_img, pr_frames_img, keypoint_names, threshold=0.05)
                    joint_aps = list(pck_results.values())[:-1]  # Exclude 'Mean' value
                    mean_ap = pck_results.get('Mean', 0.0)
                
                per_image_results[img_key] = {
                    'image_path': img_key,
                    'mAP': mean_ap,
                    'joint_APs': joint_aps,
                    'has_predictions': True,
                    'num_gt_poses': len(gt_frames_img),
                    'num_pred_poses': len(pr_frames_img)
                }
                
            except Exception as e:
                logger.error(f"Error calculating mAP for image {img_filename}: {str(e)}")
                per_image_results[img_key] = {
                    'image_path': img_key,
                    'mAP': 0.0,
                    'joint_APs': [0.0] * self.num_joints,
                    'has_predictions': False,
                    'error': str(e)
                }
        
        # Add summary statistics
        valid_maps = [result['mAP'] for result in per_image_results.values() 
                     if result.get('has_predictions', False) and result['mAP'] > 0]
        
        summary = {
            'total_images': len(per_image_results),
            'images_with_predictions': len([r for r in per_image_results.values() if r.get('has_predictions', False)]),
            'mean_mAP': np.mean(valid_maps) if valid_maps else 0.0,
            'std_mAP': np.std(valid_maps) if valid_maps else 0.0,
            'min_mAP': np.min(valid_maps) if valid_maps else 0.0,
            'max_mAP': np.max(valid_maps) if valid_maps else 0.0,
            'dataset_name': self.dataset_name
        }
        
        result = {
            'summary': summary,
            'per_image_results': per_image_results
        }
        
        logger.info(f"=> Per-image mAP calculation completed. Mean mAP: {summary['mean_mAP']:.4f}")
        
        return result
