from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/got10k_lmdb'
    settings.got10k_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/lasot_lmdb'
    settings.lasot_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/lasot'
    settings.network_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/nfs'
    settings.otb_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/otb'
    settings.prj_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack'
    settings.result_plot_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/test/result_plots'
    settings.results_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/test/tracking_results'    # Where to store tracking results
    settings.rgbdt_test_dir = '/data_A/xuefeng/RGBDT500/Test'
    settings.save_dir = '/data_A/xuefeng/2025NeurIPS/RDTTrack'
    settings.segmentation_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/test/segmentation_results'
    settings.tc128_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/trackingnet'
    settings.uav_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/uav'
    settings.vot18_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/vot2018'
    settings.vot22_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/vot2022'
    settings.vot_path = '/data_A/xuefeng/2025NeurIPS/RDTTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

