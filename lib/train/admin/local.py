class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data_A/xuefeng/2025NeurIPS/RDTTrack/pretrained_networks'
        self.got10k_val_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/coco_lmdb'
        self.coco_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/coco'
        self.lasot_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/lasot'
        self.got10k_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/got10k/train'
        self.trackingnet_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/trackingnet'
        self.depthtrack_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/depthtrack/train'
        self.lasher_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/lasher/trainingset'
        self.visevent_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/visevent/train'
        self.rgbdt_dir = '/data_A/xuefeng/RGBDT500/Train_400'
