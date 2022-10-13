import cv2
import os
import numpy as np
import argparse
from utils.helper import create_folder, remove_files

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='./data/COWC/Toronto_03553.7.8')
parser.add_argument("--loc_idx", type=str, default='Toronto_03553.7.8')
parser.add_argument("--img_type", type=str, default='.jpg')
parser.add_argument("--mask_name", type=str, default=None)
parser.add_argument("--obj_name", type=str, default='car')
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--win_size", type=int, default=40)
args = parser.parse_args()

# save the cropped images within the region in temp folder
unlabeled_imgs_path = './temp'

# the directory of the dataset, such as xView, DIOR, COWC
data_dir = args.data_dir
# the location index or name
loc_idx = args.loc_idx
image_name = loc_idx + args.img_type
# the region-level mask
mask_name = None
obj_name = args.obj_name

# the sliding window size and stride to crop the images within the region-of-interest
stride = args.stride
win_size = args.win_size

image = cv2.imread(os.path.join(data_dir, image_name))
if mask_name == None:
    mask = np.ones((image.shape[0],image.shape[1]))
else:
    mask = cv2.imread(os.path.join(data_path, mask_name+'_label.jpeg'), 0) / 255

print('the shape of image and mask: ', image.shape, mask.shape)

# create a folder to save the cropped images
#if the folder exists, remove the existing files in the folder
create_folder(unlabeled_imgs_path)
remove_files(unlabeled_imgs_path)

num_total_img = 0
img_names = []
for row in range(0,mask.shape[0]-win_size,stride):
    for col in range(0,mask.shape[1]-win_size,stride):
        sub_mask = mask[row:row+win_size, col:col+win_size]
        if np.count_nonzero(sub_mask) == win_size*win_size:
            sub_img = image[row:row+win_size, col:col+win_size]
            cv2.imwrite(os.path.join(unlabeled_imgs_path, str(row)+'_'+str(col)+'.png'), sub_img)
            num_total_img += 1
            img_names.append(str(row)+'_'+str(col))
print('done to save cropped images in ', unlabeled_imgs_path)
