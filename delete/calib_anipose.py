import numpy as np
from aniposelib.boards import DoubleSidedCharucoBoard
from aniposelib.cameras import CameraGroup

vidnames = [['calibration_videos/sba_cam08150951.avi'],
            ['calibration_videos/sba_cam08151951.avi'],
            ['calibration_videos/sba_cam08152151.avi'],
            ['calibration_videos/sba_cam08154551.avi',
             # 'calibration_videos/extrinsic_08150951-08151951_cam08150951.avi',
             # 'calibration_videos/extrinsic_08150951-08151951_cam08151951.avi',
             # 'calibration_videos/extrinsic_08150951-08152151_cam08150951.avi',
             # 'calibration_videos/extrinsic_08150951-08152151_cam08152151.avi',
             # 'calibration_videos/extrinsic_08151951-08152151_cam08151951.avi',
             # 'calibration_videos/extrinsic_08151951-08152151_cam08152151.avi',
             # 'calibration_videos/extrinsic_08154551-08150951_cam08150951.avi',
             # 'calibration_videos/extrinsic_08154551-08150951_cam08154551.avi',
             # 'calibration_videos/extrinsic_08154551-08151951_cam08151951.avi',
             # 'calibration_videos/extrinsic_08154551-08151951_cam08154551.avi',
             # 'calibration_videos/extrinsic_08154551-08152151_cam08152151.avi',
             # 'calibration_videos/extrinsic_08154551-08152151_cam08154551.avi'
             ]]

cam_names = ['08154551', '08150951', '08151951', '08152151']

n_cams = len(vidnames)

board = DoubleSidedCharucoBoard(10, 7,
                                square_length=30,  # here, in mm but any unit works
                                marker_length=20,
                                marker_bits=4, dict_size=100)

cgroup = CameraGroup.from_names(cam_names, fisheye=False)

cgroup.calibrate_videos(vidnames, board)

cgroup.dump('calibration.toml')