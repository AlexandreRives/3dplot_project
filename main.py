import threedplot
import pickle
import os

if __name__ == '__main__':
    params = os.getcwd() + "/params/"
    print("Enter the trial number : (ex. 001)")
    trial_num = str(input())
    # print("Set the paths to launch the process")
    # print("2d coordinates csv folder :")
    # csv_folder = path + "/coords"
    # print("Name of the parameters folder :")
    # param_folder = str(input())
    with open(params + 'extrinsic_params.pickle', 'rb') as extrinsic_file:
        extrinsic_params = pickle.load(extrinsic_file)
    with open(params + 'intrinsic_params.pickle', 'rb') as intrinsic_file:
        intrinsic_params = pickle.load(intrinsic_file)
    with open(params + 'rectify_params.pickle', 'rb') as rectify_file:
        rectify_params = pickle.load(rectify_file)
    print("Generating 3d coordinates csv file and 3d animation video...")
    threedplot.generate_coordinates_and_video(extrinsic_params, intrinsic_params, rectify_params, trial_num)
    print("Success !")
    print("See the files into the videos and coordinates folders")