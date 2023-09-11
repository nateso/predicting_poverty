import torch
import os

# prints out the training device to check whether GPU is available
# training device
print("Hi there! This is a test file to check whether GPU is available.")
print("\n")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training device: {device}")
print("="*100)
print('\n')
print('Now check whether I can access files in the data folder')
print('\n')

# check whether I can access files in the data folder

# set the global file paths
root_data_dir = "../../../scratch/users/nschmid5/data_analysis/Data"
#root_data_dir = "../../Data"

# the lsms data
lsms_pth = f"{root_data_dir}/lsms/processed/labels_cluster_v1.csv"

# the feature data (OSM + precipitation)
feat_data_pth = f"{root_data_dir}/feature_data/tabular_data.csv"

# set the image directories
sat_img_dir = f"{root_data_dir}/satellite_imgs"

# list all files in the delta image folder:
files = os.listdir(f"{sat_img_dir}/RS_v2/RS_v2_delta")
print(f"Number of files in the delta folder: {len(files)}")
print('\n')
print("="*100)

print("Over and out!")


