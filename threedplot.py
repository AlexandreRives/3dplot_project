import json
import sys

import pandas as pd
import numpy as np
import os
import cv2
import glob
import re
import pickle
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from scipy import linalg

from combinedVideoCv2 import merge_videos
from low_pass_filter import low_pass_filter

def locate_dlt(cam_sns, camera_coords, intrinsic_params, extrinsic_params, rectify_params=None):
    A = []
    cameras_used = 0
    location = np.zeros((1, 3))
    for idx in range(len(cam_sns)):
        sn = cam_sns[idx]
        if len(camera_coords[sn]) > 0:
            point = camera_coords[sn]
            RT = np.concatenate([extrinsic_params[sn]['r'], extrinsic_params[sn]['t']], axis=-1)
            P = intrinsic_params[sn]['k'] @ RT
            A.append(point[1] * P[2, :] - P[1, :])
            A.append(P[0, :] - point[0] * P[2, :])
            cameras_used = cameras_used + 1
    if cameras_used > 1:
        A = np.array(A)
        A = np.array(A).reshape((cameras_used * 2, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices=False)

        location = Vh[-1, 0:3] / Vh[-1, 3]
        location = np.reshape(location, (1, 3))

        # Apply rectification
        if rectify_params is not None:
            table_center = rectify_params['origin']
            v1=rectify_params['x_axis'] - table_center
            v2=rectify_params['y_axis'] - table_center
            v3=rectify_params['z_axis'] - table_center
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            v3 = v3 / np.linalg.norm(v3)
            M_inv=np.linalg.inv(np.transpose(np.squeeze([v1,v2,v3])))
            location=np.transpose(np.matmul(M_inv,np.transpose((location-table_center))))
    return [location, cameras_used]

def locate_3d_function(body_part, cam_df, intrinsic_params, extrinsic_params, rectify_params):
    """
    Triangulate the 3d coordinates with 2d coordinates dataframes

    Parameters :
    --------------
    body_part : List
        List of bodyparts

    cam : List
        List of dataframes

    cam_sns : List
        List of camera numbers
    
    intrinsic_params : Dictionary
        intrinsic parameters

    extrinsic_params : Dictionary
        extrinsic parameters

    rectify_params : Dictionary
        rectify parameters
    """
    cam_sns=list(cam_df.keys())
    # table to store the 3d coordinates
    result = []

    # create the bodyparts dictionnary
    n_frames = len(cam_df[cam_sns[0]])
    for frame_idx in range(n_frames):
        bodypart_dict = {}
        for cam_sn in cam_sns:
            # Put a threshold
            scorer=cam_df[cam_sn].columns.levels[0][0]

            if cam_df[cam_sn][(scorer,body_part,'likelihood')][frame_idx] > 0.15:
                bodypart_dict[cam_sn] = np.array([cam_df[cam_sn][(scorer,body_part,'x')][frame_idx],
                                                  cam_df[cam_sn][(scorer,body_part,'y')][frame_idx]])
            else:
                bodypart_dict[cam_sn] = []

                # call the function to compute the 3d points and add it to the table
        coord, n_cams = locate_dlt(cam_sns=cam_sns, camera_coords=bodypart_dict,
                                      intrinsic_params=intrinsic_params,
                                      extrinsic_params=extrinsic_params,
                                      rectify_params=rectify_params)
        if n_cams < 2:
            coord = coord*float('NaN')
        result.append(coord)

    return result

def generate_coordinates_and_video(settings, extrinsic_params, intrinsic_params, rectify_params, video_folder, output_folder):
    """
    Save the 3d coordinates in a csv file and build the 3d animation video

    Parameters :
    --------------
    extrinsic_params : Dictionary
        extrinsic parameters

    intrinsic_params : Dictionary
        intrinsic parameters

    rectify_params : Dictionary
        rectify parameters

    trial_num : String
        trial number
    """

    cam_sns = settings['cam_sns']

    # use glob to get all the csv files in the folder
    h5_files = glob.glob(os.path.join(video_folder, "*filtered.h5"))
    cam_df = {}
    # loop over the list of csv files
    for f in sorted(h5_files):
        # read the csv file
        df = pd.read_hdf(f)
        cam_idx=int(os.path.split(f)[1].split('_')[1][3:])
        cam_sn=cam_sns[cam_idx]

        cam_df[cam_sn]=df

    # table to store all the coordinates from all the bodyparts
    coordinates = {}
    body_parts = list(cam_df[cam_sns[0]].columns.levels[1])

    final_coord = {}
    for body_part in body_parts:
        filtered_coord = []
        coordinates[body_part] = locate_3d_function(body_part, cam_df, intrinsic_params, extrinsic_params,
                                                    rectify_params)

        # Process with low pass filter.
        # Low pass on X, Y, Z
        bodypart_coord = np.row_stack(coordinates[body_part])
        for idx in range(0, 3):
            coord = low_pass_filter(bodypart_coord[:, idx], 200, 10)
            filtered_coord.append(coord)

        filtered_coord = np.array(filtered_coord).T
        final_coord[body_part] = filtered_coord

    skeleton = [
        ['shoulder', 'elbow'],
        ['elbow', 'wrist'],
        ['wrist', 'thumb1'], ['thumb1', 'thumb2'], ['thumb2', 'thumb3'],
        ['wrist', 'index1'], ['index1', 'index2'], ['index2', 'index3'], ['index3', 'index4'],
        ['wrist', 'middle1'], ['middle1', 'middle2'], ['middle2', 'middle3'], ['middle3', 'middle4'],
        ['wrist', 'ring1'], ['ring1', 'ring2'], ['ring2', 'ring3'], ['ring3', 'ring4'],
        ['wrist', 'little1'], ['little1', 'little2'], ['little2', 'little3'], ['little3', 'little4']
    ]

    n_frames = len(cam_df[cam_sns[0]])

    # Colors
    norm = mcolors.Normalize(vmin=0, vmax=len(body_parts)-1)
    colors = []
    for b in range(len(body_parts)):
        colors.append(cm.cool(norm(b)))

    frames = []
    markers = []

    for frame in range(n_frames):
        for marker in final_coord:
            markers.append(marker)
            frames.append(final_coord[marker][frame])

    frames = np.asarray(frames)
    markers = np.asarray(markers)

    frames_num = np.array([np.ones(len(body_parts)) * i for i in range(n_frames)]).flatten()
    coordinates = pd.DataFrame({"frame": frames_num, "x": frames[:, 0], "y": frames[:, 1], "z": frames[:, 2], "marker": markers})

    coordinates_path = os.path.join(output_folder, '3d_coordinates.csv')
    coordinates.to_csv(coordinates_path)

    #####################
    #       Plot        #
    #####################

    coordinates_min_x_nan = min(coordinates[~np.isnan(coordinates['x'])]['x'])
    coordinates_min_y_nan = min(coordinates[~np.isnan(coordinates['y'])]['y'])
    coordinates_min_z_nan = min(coordinates[~np.isnan(coordinates['z'])]['z'])

    coordinates_max_x_nan = max(coordinates[~np.isnan(coordinates['x'])]['x'])
    coordinates_max_y_nan = max(coordinates[~np.isnan(coordinates['y'])]['y'])
    coordinates_max_z_nan = max(coordinates[~np.isnan(coordinates['z'])]['z'])

    video_file = os.path.join(output_folder, 'animation.avi')
    video = cv2.VideoWriter(filename=video_file,  # Provide a file to write the video to
                            fourcc=cv2.VideoWriter_fourcc(*'XVID'),  # Use whichever codec works for you...
                            fps=100,  # How many frames do you want to display per second in your video?
                            frameSize=(640, 480))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(coordinates_min_x_nan - 0.05, coordinates_max_x_nan + 0.05)
    ax.set_ylim(coordinates_min_y_nan - 0.05, coordinates_max_y_nan + 0.05)
    ax.set_zlim(coordinates_min_z_nan - 0.05, coordinates_max_z_nan + 0.05)

    # Hide grid lines
    ax.grid(visible=None)

    # Hide axes ticks
    ax.axis('off')

    # ax.view_init(90, 0)

    # Loop over frames
    for frame in range(0, n_frames):
        data = coordinates[coordinates['frame'] == frame]

        # Draw frame
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(coordinates_min_x_nan - 0.05, coordinates_max_x_nan + 0.05)
        ax.set_ylim(coordinates_min_y_nan - 0.05, coordinates_max_y_nan + 0.05)
        ax.set_zlim(coordinates_min_z_nan - 0.05, coordinates_max_z_nan + 0.05)

        # Hide grid lines
        plt.grid(visible=None)

        # Hide axes ticks
        ax.axis('off')

        # ax.view_init(90, 0)

        # Loop over all body parts
        for b_idx,body_part in enumerate(data['marker']):

             # If not coordinate is NaN
             x = data[data['marker'] == body_part]['x']
             y = data[data['marker'] == body_part]['y']
             z = data[data['marker'] == body_part]['z']
             if not(np.isnan(x.iloc[0]) or np.isnan(y.iloc[0]) or np.isnan(z.iloc[0])):
                # Get color for that body part
                graph = ax.scatter(x, y, z, s=10, color=colors[b_idx])

        # Loop through skeleton
        for bp1, bp2 in skeleton:
            # If neither coordinate is NaN
            data1 = data[data['marker'] == bp1]
            data2 = data[data['marker'] == bp2]
            if data1['x'].iloc[0] != np.float64('NaN') and data1['y'].iloc[0] != np.float64('NaN') and data1['z'].iloc[0] != np.float64('NaN') and data2['x'].iloc[0] != np.float64('NaN') and data2['y'].iloc[0] != np.float64('NaN') and data2['z'].iloc[0] != np.float64('NaN'):
                # Draw line
                line = ax.plot([float(data1.x), float(data2.x)], [float(data1.y), float(data2.y)],
                               [float(data1.z), float(data2.z)], color='gray')

        # Create filename
        #fig.savefig("dataname_{:03}.png".format(frame))
        # redraw the canvas
        fig.canvas.draw()
        # convert canvas to image
        plt_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

        video.write(plt_img)

    cv2.destroyAllWindows()
    video.release()

    video_files = glob.glob(os.path.join(video_folder, "*filtered_labeled.mp4"))
    video_files.append(video_file)

    combined_video = os.path.join(output_folder, 'video.avi')

    #check avi_files
    merge_videos(video_files,
        combined_video,
        grid_size=(3, 2))

# joints = ['elbow', 'wrist',
#           'thumb1', 'thumb2', 'thumb3',
#           'index1', 'index2', 'index3', 'index4',
#           'middle1', 'middle2', 'middle3', 'middle4',
#           'ring1', 'ring2', 'ring3', 'ring4',
#           'little1', 'little2', 'little3', 'little4']

# xs = []
# ys = []
# zs = []
# for joint in joints:
#     data1 = df[(df['frame'] == 0) & (df['bodypart'] == joint)]
#     if np.isnan(float(data1.x)):
#         xs.append(0.0)
#         ys.append(0.0)
#         zs.append(0.0)
#     else:
#         xs.append(float(data1.x))
#         ys.append(float(data1.y))
#         zs.append(float(data1.z))

# lines = []
# for bp1, bp2 in skeleton:
#     data1 = df[(df['frame'] == 0) & (df['bodypart'] == bp1)]
#     data2 = df[(df['frame'] == 0) & (df['bodypart'] == bp2)]
#     line = ax.plot([float(data1.x), float(data2.x)], [float(data1.y), float(data2.y)], [float(data1.z), float(data2.z)],
#                    color='gray')
#     lines.append(line[0])

# graph = ax.scatter(xs, ys, zs, s=10, c=colors)

# ani = matplotlib.animation.FuncAnimation(fig, update_graph, n_frames, interval=1, blit=False, fargs=[lines])
# plt.show()
# ani.save('3d_plot.mp4', fps=200)


# def update_graph(num, lines):
#     data = df[df['frame'] == num]
#     graph._offsets3d = (data.x, data.y, data.z)
#     for line, (bp1, bp2) in zip(lines, skeleton):
#         data1 = df[(df['frame'] == num) & (df['bodypart'] == bp1)]
#         data2 = df[(df['frame'] == num) & (df['bodypart'] == bp2)]
#         if len(data1) and len(data2):
#             line._verts3d = [float(data1.x), float(data2.x)], [float(data1.y), float(data2.y)], [float(data1.z),
#                                                                                                  float(data2.z)]
#     return lines


# Process with Kalman Filter
# corrected = []
# for idx in range(len(coordinates[body_part])):
#
#     coordinate = coordinates[body_part][idx][0]
#     if not np.any(np.isnan(coordinate)):
#         if not initialized:
#             k.initKalman(coordinate[0], coordinate[1], coordinate[2])
#             initialized = True
#             corrected.append(coordinate)
#         else:
#             p = k.kalmanPredict()
#             s = k.kalmanCorrect(coordinate[0], coordinate[1], coordinate[2])
#             corrected.append(s)
#     else:
#         corrected.append(coordinate)
# corrected = np.vstack(corrected)
#
# for i in range(3):
#     corrected[:, i] = k.fill_nan(corrected[:, i])
# coordinates[body_part] = corrected

if __name__ == '__main__':

    try:
        calib_folder = sys.argv[1]
        print('USING: %s' % calib_folder)
    except:
        calib_folder = None

    try:
        video_folder = sys.argv[2]
        print('USING: %s' % video_folder)
    except:
        video_folder = None

    try:
        output_folder = sys.argv[3]
        print('USING: %s' % output_folder)
    except:
        output_folder = None

    # opening a json file
    with open(os.path.join(calib_folder, 'settings.json')) as settings_file:
        settings = json.load(settings_file)

    with open(os.path.join(calib_folder, 'extrinsic_params.pickle'), 'rb') as extrinsic_file:
        extrinsic_params = pickle.load(extrinsic_file)
    with open(os.path.join(calib_folder, 'intrinsic_params.pickle'), 'rb') as intrinsic_file:
        intrinsic_params = pickle.load(intrinsic_file)
    with open(os.path.join(calib_folder, 'rectify_params.pickle'), 'rb') as rectify_file:
        rectify_params = pickle.load(rectify_file)
    print("Generating 3d coordinates csv file and 3d animation video...")
    generate_coordinates_and_video(settings, extrinsic_params, intrinsic_params, rectify_params, video_folder,
                                   output_folder)
    print("Success !")
    print("See the files into the videos and coordinates folders")