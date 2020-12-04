import cv2
import os
import numpy as np
from utils import create_folder, remove_files

# save the cropped images within the region in temp folder
unlabeled_imgs_path = './temp'

# the directory of the dataset, such as xView, DIOR, COWC
data_dir = '/data/weiweidu/COWC/DetectionPatches_800x800/Toronto_ISPRS'
# the location index or name
loc_idx = 'Toronto_03559.8.2'
image_name = loc_idx+'.jpg'
# the region-level mask
mask_name = None
obj_name = 'car'

# the sliding window size and stride to crop the images within the region-of-interest
stride = 20
win_size = 40

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
print('done to write cropped images in ', unlabeled_imgs_path)
