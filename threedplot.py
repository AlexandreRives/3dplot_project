import calib_tools as ct
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
import kalman as k
from low_pass_filter import low_pass_filter

def create_df(csv_files):
    # list of all the dataframes coming from the csv_files
    df_list = list()

    # loop over the list of csv files
    for f in sorted(csv_files):
        # read the csv file
        df = pd.read_csv(f, skiprows=2)
        bodyparts_name = ["elbow_x", "elbow_y", "likelihood_elbow", "wrist_x", "wrist_y", "likelihood_wrist",
                          "thumb1_x", "thumb1_y", "likelihood_thumb1",
                          "thumb2_x", "thumb2_y", "likelihood_thumb2", "thumb3_x", "thumb3_y", "likelihood_thumb3",
                          "index1_x", "index1_y", "likelihood_index1",
                          "index2_x", "index2_y", "likelihood_index2", "index3_x", "index3_y", "likelihood_index3",
                          "index4_x", "index4_y", "likelihood_index4",
                          "middle1_x", "middle1_y", "likelihood_middle1", "middle2_x", "middle2_y",
                          "likelihood_middle2", "middle3_x", "middle3_y", "likelihood_middle3",
                          "middle4_x", "middle4_y", "likelihood_middle4", "ring1_x", "ring1_y", "likelihood_ring1",
                          "ring2_x", "ring2_y", "likelihood_ring2",
                          "ring3_x", "ring3_y", "likelihood_ring3", "ring4_x", "ring4_y", "likelihood_ring4",
                          "little1_x", "little1_y", "likelihood_little1",
                          "little2_x", "little2_y", "likelihood_little2", "little3_x", "little3_y",
                          "likelihood_little3", "little4_x", "little4_y", "likelihood_little4"]

        df.drop(df.columns[0], axis=1, inplace=True)

        df.columns = bodyparts_name

        # keep the cam* from the csv_file and add the cam_sns column into the df.
        print('File Name:', f.split("/")[-1])

        df_list.append(df)

    return df_list


def locate_3d_function(body_part, cam, cam_sns, intrinsic_params, extrinsic_params, rectify_params):
    # table to store the 3d coordinates
    result = []

    # create the bodyparts dictionnary
    n_frames = len(cam[0])
    for frame_idx in range(n_frames):
        bodypart_dict = {}
        for cam_idx in range(len(cam)):
            # Put a threshold
            if cam[cam_idx]['likelihood_{}'.format(body_part)][frame_idx] > 0.15:
                bodypart_dict[cam_sns[cam_idx]] = np.array([cam[cam_idx]['{}_x'.format(body_part)][frame_idx],
                                                            cam[cam_idx]['{}_y'.format(body_part)][frame_idx]])

        # Check the cameras if more than X
        sn = list(bodypart_dict.keys())
        # call the function to compute the 3d points and add it to the table
        coord, n_cams = ct.locate_dlt(cam_sns=sn, camera_coords=bodypart_dict,
                                      intrinsic_params=intrinsic_params,
                                      extrinsic_params=extrinsic_params,
                                      rectify_params=rectify_params)
        if n_cams < 2:
            coord = coord*float('NaN')
        result.append(coord)

    return result

##############################
#     Generate 3D points     #
##############################

def generate_coordinates(csv_path, extrinsic_params, intrinsic_params, rectify_params):
    # cameras serial numbers
    cam_sns = ['08154551', '08150951', '08151951', '08152151']

    # use glob to get all the csv files in the folder
    csv_files = glob.glob(os.path.join(csv_path, "*.csv"))

    # call the function to get all the csv_file and store it into a list of df
    cam = create_df(csv_files)

    # table to store all the coordinates from all the bodyparts
    coordinates = {}
    body_parts = list(cam[0].columns[1:])
    body_parts = list(pd.unique([x.split('_')[0] for x in body_parts]))
    body_parts.remove('likelihood')

    final_coord = {}
    for body_part in body_parts:
        filtered_coord = []
        coordinates[body_part] = locate_3d_function(body_part, cam, cam_sns, intrinsic_params, extrinsic_params,
                                                    rectify_params)

        # Process with low pass filter.
        # Low pass on X, Y, Z
        bodypart_coord = np.row_stack(coordinates[body_part])
        for idx in range(0, 3):
            coord = low_pass_filter(bodypart_coord[:, idx])
            filtered_coord.append(coord)

        filtered_coord = np.array(filtered_coord).T
        final_coord[body_part] = filtered_coord

    skeleton = [
        ['elbow', 'wrist'],
        ['wrist', 'thumb1'], ['thumb1', 'thumb2'], ['thumb2', 'thumb3'],
        ['wrist', 'index1'], ['index1', 'index2'], ['index2', 'index3'], ['index3', 'index4'],
        ['wrist', 'middle1'], ['middle1', 'middle2'], ['middle2', 'middle3'], ['middle3', 'middle4'],
        ['wrist', 'ring1'], ['ring1', 'ring2'], ['ring2', 'ring3'], ['ring3', 'ring4'],
        ['wrist', 'little1'], ['little1', 'little2'], ['little2', 'little3'], ['little3', 'little4']
    ]

    n_frames = len(cam[0])

    # Colors
    norm = mcolors.Normalize(vmin=0, vmax=20)
    colors = []
    for b in range(21):
        colors.append(cm.cool(norm(b)))

    frames = []
    body_parts = []

    for frame in range(n_frames):
        for body_part in final_coord:
            body_parts.append(body_part)
            # if len(coordinates[body_part][frame]) <= 1:
            frames.append(final_coord[body_part][frame])
            # else:
            #     frames.append(coordinates[body_part][frame])

    frames = np.asarray(frames)
    hand = np.asarray(body_parts)

    frames_num = np.array([np.ones(21) * i for i in range(n_frames)]).flatten()
    coordinates = pd.DataFrame({"frame": frames_num, "x": frames[:, 0], "y": frames[:, 1], "z": frames[:, 2], "bodypart": hand})

    file_exists = os.path.exists('filtered_coordinates.csv')
    if not file_exists:
        coordinates.to_csv('filtered_coordinates.csv')

    #####################
    #       Plot        #
    #####################

    coordinates_min_x_nan = min(coordinates[~np.isnan(coordinates['x'])]['x'])
    coordinates_min_y_nan = min(coordinates[~np.isnan(coordinates['y'])]['y'])
    coordinates_min_z_nan = min(coordinates[~np.isnan(coordinates['z'])]['z'])

    coordinates_max_x_nan = max(coordinates[~np.isnan(coordinates['x'])]['x'])
    coordinates_max_y_nan = max(coordinates[~np.isnan(coordinates['y'])]['y'])
    coordinates_max_z_nan = max(coordinates[~np.isnan(coordinates['z'])]['z'])

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
    for frame in range(0, len(cam[0])):
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
        for b_idx,body_part in enumerate(data['bodypart']):

             # If not coordinate is NaN
             x = data[data['bodypart'] == body_part]['x']
             y = data[data['bodypart'] == body_part]['y']
             z = data[data['bodypart'] == body_part]['z']
             if not(np.isnan(x.iloc[0]) or np.isnan(y.iloc[0]) or np.isnan(z.iloc[0])):
                # Get color for that body part
                graph = ax.scatter(x, y, z, s=10, color=colors[b_idx])

        # Loop through skeleton
        for bp1, bp2 in skeleton:
            # If neither coordinate is NaN
            data1 = data[data['bodypart'] == bp1]
            data2 = data[data['bodypart'] == bp2]
            if data1['x'].iloc[0] != np.float64('NaN') and data1['y'].iloc[0] != np.float64('NaN') and data1['z'].iloc[0] != np.float64('NaN') and data2['x'].iloc[0] != np.float64('NaN') and data2['y'].iloc[0] != np.float64('NaN') and data2['z'].iloc[0] != np.float64('NaN'):
                # Draw line
                line = ax.plot([float(data1.x), float(data2.x)], [float(data1.y), float(data2.y)],
                               [float(data1.z), float(data2.z)], color='gray')

        # Create filename
        fig.savefig("saved_frames/dataname_{:03}.png".format(frame))

    ##############################
    #       Images to Video      #
    ##############################

    # Put frames together into video
    image_folder = 'saved_frames'
    video_name = 'video.avi'

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # video = cv2.VideoWriter(video_name, 0, 1, (width,height))
    video = cv2.VideoWriter(filename=video_name,  #Provide a file to write the video to
        fourcc=cv2.VideoWriter_fourcc(*'XVID'),           #Use whichever codec works for you...
        fps=100,                                        #How many frames do you want to display per second in your video?
        frameSize=(width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    path = os.getcwd() + '/saved_frames'
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

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