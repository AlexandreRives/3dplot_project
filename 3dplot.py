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
import matplotlib.animation
import matplotlib.colors as mcolors


def create_df(csv_files, cam_sns):
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
        res = re.findall("cam(\d+)", f)
        #df.insert(loc=0, column='cam_sns', value=cam_sns[int(res[0])])

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
            if cam[cam_idx]['likelihood_{}'.format(body_part)][frame_idx] > 0.01:
                bodypart_dict[cam_sns[cam_idx]] = np.array([cam[cam_idx]['{}_x'.format(body_part)][frame_idx],
                                                            cam[cam_idx]['{}_y'.format(body_part)][frame_idx]])

        # Check the cameras if more than X
        sn = list(bodypart_dict.keys())
        # call the function to compute the 3d points and add it to the table
        coord, n_cams = ct.locate_sba(cam_sns=sn, camera_coords=bodypart_dict,
                                      intrinsic_params=intrinsic_params,
                                      extrinsic_params=extrinsic_params,
                                      rectify_params=rectify_params)
        if n_cams<4:
            coord=coord*float('NaN')
        result.append(coord)

    return result


#####################
#   Params files    #
#####################

with open('/home/arives/PycharmProjects/3dplot_project/params/extrinsic_sba_params.pickle', 'rb') as pickle_file:
    extrinsic_params = pickle.load(pickle_file)
# with open('/home/arives/PycharmProjects/3dplot_project/params/sba_params_new.pickle', 'rb') as pickle_file:
#     extrinsic_params = pickle.load(pickle_file)
with open('/home/arives/PycharmProjects/3dplot_project/params/intrinsic_params.pickle', 'rb') as pickle_file:
    intrinsic_params = pickle.load(pickle_file)
with open('/home/arives/PycharmProjects/3dplot_project/params/rectify_params.pickle', 'rb') as pickle_file:
    rectify_params = pickle.load(pickle_file)

##############################
#     Generate 3D points     #
##############################

# cameras serial numbers
cam_sns = ['08154551', '08150951', '08151951', '08152151']

# use glob to get all the csv files in the folder
csv_files = glob.glob(os.path.join("/home/arives/PycharmProjects/3dplot_project/coords/", "*.csv"))

# call the function to get all the csv_file and store it into a list of df
cam = create_df(csv_files, cam_sns)

# table to store all the coordinates from all the bodyparts
coordinates = {}
body_parts = list(cam[0].columns[1:])
body_parts = list(pd.unique([x.split('_')[0] for x in body_parts]))
body_parts.remove('likelihood')

for body_part in body_parts:
    coordinates[body_part] = locate_3d_function(body_part, cam, cam_sns, intrinsic_params, extrinsic_params,
                                                rectify_params)


#####################
#       Plot        #
#####################

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


joints = ['elbow', 'wrist',
          'thumb1', 'thumb2', 'thumb3',
          'index1', 'index2', 'index3', 'index4',
          'middle1', 'middle2', 'middle3', 'middle4',
          'ring1', 'ring2', 'ring3', 'ring4',
          'little1', 'little2', 'little3', 'little4']

skeleton = [
    ['elbow', 'wrist'],
    ['wrist', 'thumb1'], ['thumb1', 'thumb2'], ['thumb2', 'thumb3'],
    ['wrist', 'index1'], ['index1', 'index2'], ['index2', 'index3'], ['index3', 'index4'],
    ['wrist', 'middle1'], ['middle1', 'middle2'], ['middle2', 'middle3'], ['middle3', 'middle4'],
    ['wrist', 'ring1'], ['ring1', 'ring2'], ['ring2', 'ring3'], ['ring3', 'ring4'],
    ['wrist', 'little1'], ['little1', 'little2'], ['little2', 'little3'], ['little3', 'little4']
]

n_frames = len(cam[0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(-0.2, 0.6)
# ax.set_ylim(-0.3, 0.1)
# ax.set_zlim(-0.4, 0.4)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Colors
norm = mcolors.Normalize(vmin=0, vmax=20)
colors = []
for b in range(21):
    colors.append(cm.cool(norm(b)))


frames = []
body_parts = []

for frame in range(n_frames):
    for body_part in coordinates:
        body_parts.append(body_part)
        # if len(coordinates[body_part][frame]) <= 1:
        frames.append(coordinates[body_part][frame][0])
        # else:
        #     frames.append(coordinates[body_part][frame])

frames = np.asarray(frames)
hand = np.asarray(body_parts)

t = np.array([np.ones(21) * i for i in range(n_frames)]).flatten()
df = pd.DataFrame({"frame": t, "x": frames[:, 0], "y": frames[:, 1], "z": frames[:, 2], "bodypart": hand})


# Loop over frames
for frame in range (0, len(cam[0])):
    data = df[df['frame'] == frame]

    # Draw frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.2, 0.6)
    ax.set_ylim(-0.3, 0.1)
    ax.set_zlim(-0.4, 0.4)
    # ax.set_xlim(min(df['x'])-0.2, max(df['x'])+0.2)
    # ax.set_ylim(min(df['y'])-0.2, max(df['y'])+0.2)
    # ax.set_zlim(min(df['z'])-0.2, max(df['z'])+0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

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
    #plt.draw()
    fig.savefig("saved_frames/dataname_{:03}.png".format(frame))

##############################
#       Images to Video        #
##############################

# Put frames together into video
image_folder = 'saved_frames'
video_name = 'video.avi'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()





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