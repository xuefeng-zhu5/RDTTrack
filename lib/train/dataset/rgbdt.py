import os
import os.path
import torch
import numpy as np
import pandas
import csv
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import cv2

def get_rgbdt_frame(color_path, depth_path, infrared_path, depth_clip=False):
    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = None

    if depth_path:
        dp = cv2.imread(depth_path, -1)

        if depth_clip:
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp > max_depth] = max_depth
    else:
        dp = None


    if infrared_path:
        infrared = cv2.imread(infrared_path, -1)
        infrared = cv2.cvtColor(infrared, cv2.COLOR_BGR2RGB)


    dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dp = np.asarray(dp, dtype=np.uint8)
    colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)  # (h,w) -> (h,w,3)
    img = cv2.merge((rgb, colormap, infrared))
    return img

class RGBDT(BaseVideoDataset):
    """ DepthTrack dataset.
    """

    def __init__(self, root=None, dtype='rgbcolormap', image_loader=jpeg4py_loader): #  vid_ids=None, split=None, data_fraction=None
        """
        args:

            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the lasot depth dataset.
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        root = env_settings().rgbdt_dir if root is None else root
        super().__init__('rgbdt', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.sequence_list = self._build_sequence_list(root)

        self.seq_per_class, self.class_list = self._build_class_list()
        self.class_list.sort()
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

    def _build_sequence_list(self, root):
        sequence_list = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        class_list = []
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('/')[0]

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class, class_list

    def get_name(self):
        return 'rgbdt'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        data = pandas.read_csv(bb_anno_file, delimiter=',', header=None,
                           dtype={0: str, 1: np.float32, 2: np.float32, 3: np.float32, 4: np.float32}, na_filter=False,
                           low_memory=False)
        frame_names = data.iloc[:, 0].values.tolist()
        gt_data = data.iloc[:, 1:5].values

        max_frame_num = int(frame_names[-1].split('.')[0])

        gt = np.zeros((max_frame_num, 4), dtype=np.float32)

        for i, frame_name in enumerate(frame_names):
            frame_num = int(frame_name.split('.')[0])
            gt[frame_num - 1] = gt_data[i]

        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)  # xywh just one kind label
        '''
        if the box is too small, it will be ignored
        '''
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        '''
        return depth image path
        '''
        return os.path.join(seq_path, 'color', '{:08}.png'.format(frame_id+1)) , os.path.join(seq_path, 'depth', '{:08}.png'.format(frame_id+1)), os.path.join(seq_path, 'infrared', '{:08}.png'.format(frame_id+1)) # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        '''
        Return :
            - colormap from depth image
            - 3xD = [depth, depth, depth], 255
            - rgbcolormap
            - rgb3d
            - color
            - raw_depth
        '''
        color_path, depth_path, infrared_path = self._get_frame_path(seq_path, frame_id)
        # if_reshape_matrix = get_infrared_matrix(seq_path, frame_id)
        img = get_rgbdt_frame(color_path, depth_path, infrared_path, depth_clip=True)
        return img

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-1]
        return raw_class
        # return self.split

    def get_class_name(self, seq_id):
        depth_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(depth_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta


