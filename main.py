import threedplot
import pickle

if __name__ == '__main__':
    print("Set the paths to launch the process")
    print("2d coordinates csv file path :")
    csv_path = str(input())
    print("Parameters path :")
    param_path = str(input())
    with open(param_path + 'extrinsic_params.pickle', 'rb') as extrinsic_file:
        extrinsic_params = pickle.load(extrinsic_file)
    with open(param_path + 'intrinsic_params.pickle', 'rb') as intrinsic_file:
        intrinsic_params = pickle.load(intrinsic_file)
    with open(param_path + 'rectify_params.pickle', 'rb') as rectify_file:
        rectify_params = pickle.load(rectify_file)
    print("Generating 3d coordinates and 3d animation...")
    threedplot.generate_coordinates_and_video(csv_path, extrinsic_params, intrinsic_params, rectify_params)
    print("Success !")
    print("See the file into your folder")