import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
import os

class RGBDTDataset(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbdt_test_dir
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        start_frame = 1
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        ground_truth_rect = ground_truth_rect.reshape(1, 4)
        color_folder = os.path.join(self.base_path, sequence_path, 'color')
        color_files = [f for f in os.listdir(color_folder) if f.endswith('.png')]
        end_frame = len(color_files)

        depth_frames = ['{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]
        color_frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.png'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]

        infrared_frames = ['{base_path}/{sequence_path}/infrared/{frame:0{nz}}.png'.format(base_path=self.base_path,
                                                                                     sequence_path=sequence_path,
                                                                                     frame=frame_num, nz=nz)for frame_num in range(start_frame, end_frame + 1)]
        frames = []
        for c_path, d_path, i_path in zip(color_frames, depth_frames, infrared_frames):
                frames.append({'color': c_path, 'depth': d_path, 'infrared': i_path})

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        return Sequence(sequence_name, frames, 'rgbdt', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [f'{i:03}' for i in range(1, 101)]
        return sequence_list
