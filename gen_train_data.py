from shutil import copyfile
import os
import argparse
import numpy as np
from utils.helper import create_folder, remove_files, load_file_names

'''
manually choose a target images in the folder ('./temp') saving the cropped images
and select all or a part of cropped images for the training phase
if the computation resources are limited, please select a part of cropped images
'''

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='COWC')
parser.add_argument("--loc_idx", type=str, default='Toronto_03553.7.8')
parser.add_argument("--obj_name", type=str, default='car')
parser.add_argument("--pos_names", type=str, default='100_100',
                   help="labeled image name, separated by comma")
args = parser.parse_args()

unlabeled_imgs_path = './temp'
# the name of chosen target images, named by top_left coordinate
pos_names = args.pos_names.split(',')
img_type = '.png'

# the dir to save manually labeled and classified positive images
dataset_name = args.dataset_name
obj_name = args.obj_name
#location name or index in the dataset
loc_idx = args.loc_idx
                    
sample_path = os.path.join('./data', dataset_name)
loc_name = loc_idx

pos_folder_path = os.path.join(sample_path, loc_name, 'positive')
subset_folder_path = os.path.join(sample_path, loc_name, 'subset')

# create a folder to save the target images, if exist, clean it first
create_folder(pos_folder_path)
remove_files(pos_folder_path)


for img_name in pos_names:
    src = os.path.join(unlabeled_imgs_path, img_name+img_type)
    dst = os.path.join(pos_folder_path, img_name+img_type)
    copyfile(src, dst)
    
num_imgs = len([name for name in os.listdir(unlabeled_imgs_path) if os.path.isfile(os.path.join(unlabeled_imgs_path, name))])


create_folder(subset_folder_path)
remove_files(subset_folder_path)

img_names = load_file_names(unlabeled_imgs_path)
img_names = np.array(img_names)
np.random.shuffle(img_names)

print('==== the labeled positive images are saved in %s ===='%(pos_folder_path))
