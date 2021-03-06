import os
import numpy as np
import cv2
from keras.backend import cast_to_floatx


def normalize_minus1_1(data):
    return 2*(data/255.) - 1

def standarization(x_train):
    mean, std = x_train.mean(), x_train.std()
    # global standardization of pixels
    pixels = (x_train - mean) / std
    # clip pixel values to [-1,1]
    pixels = np.clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    return pixels


def load_wetland_samples(target_data_path,img_size=28,channel=3, resize=False):
    x_train, f_name = [], []
    for root, directory, files in os.walk(target_data_path):
        for fname in files:
            if '.png' not in fname and '.jpg' not in fname:
                continue
            if channel == 1:
                img = cv2.imread(os.path.join(root, fname),0)
                if img is None:
                    return np.zeros((1, img_size, img_size)), None
                if img_size < 48 and resize==True:
                    img = cv2.resize(img, (48,48))
                x_train.append(img)
            else:
                img = cv2.imread(os.path.join(root, fname))
                if img is None:
                    return np.zeros((1, img_size, img_size, 3)), None
                if img_size < 48 and resize==True:
                    img = cv2.resize(img, (48,48))
                x_train.append(img.astype('float64'))
            f_name.append(fname)
    x_train = np.array(x_train) 
    #x_train = normalize_minus1_1(x_train)
   # y_train = np.array([[0]]*x_train.shape[0])
    #np.random.shuffle(x_train)
    return x_train, f_name
    
def load_all_data(map_path, mask_path, img_size, stride, flip=False, resize=False):
    x_test, img_name = [], []
    map_img = cv2.imread(map_path)
    #map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path,0)
    #print(map_img.shape, mask.shape)
    if flip == True:
        mask = 255 - mask
    for i in range(0, map_img.shape[0]-img_size, stride):
        for j in range(0, map_img.shape[1]-img_size, stride):
            if mask_path != '':
                cropped_img = mask[i:i+img_size, j:j+img_size]
                nums = np.where(cropped_img==255)[0].shape[0]
                if nums == img_size*img_size:
                    img = map_img[i:i+img_size, j:j+img_size]
                    if img_size < 48 and resize==True:
                        img = cv2.resize(img, (48,48))
                    x_test.append(img)
                    img_name.append(str(i)+'_'+str(j))
            else:
                img = map_img[i:i+img_size, j:j+img_size]
                if img.shape != (img_size,img_size,3):
                    continue
                if img_size < 48 and resize==True:
                    img = cv2.resize(img, (48,48))
                x_test.append(img)
                img_name.append(str(i)+'_'+str(j))
    x_test = np.array(x_test) 
    #x_test = normalize_minus1_1(x_test)
   # y_test = np.array([[0]]*x_test.shape[0]) 
    #np.random.shuffle(x_test)
    return x_test, img_name


def padding_data_helper(data, nums):
    if abs(data.shape[0]-nums) <= data.shape[0]:
        data = np.vstack((data, data[:abs(data.shape[0]-nums)]))
    else:
        num_repeat = int(nums/data.shape[0])
        data = np.repeat(data, num_repeat+1, axis=0)
        print('padding nums: ', data.shape, abs(data.shape[0]-nums))
        data = np.vstack((data, data[:abs(data.shape[0]-nums)]))
        print('data shape: ', data.shape)
    return data
    
def padding_data(target_samples, all_samples):
    max_num = max(target_samples.shape[0], all_samples.shape[0])
    if max_num % 100 <= 50:
        nums = max_num - max_num % 100
    else:
        nums = max_num + (100-max_num % 100)
    print('the nums of samples: ', nums)
    if target_samples.shape[0] >= nums:
        target_samples = target_samples[:nums]
    else:
        target_samples = padding_data_helper(target_samples, nums)
    if all_samples.shape[0] >= nums:
        all_samples = all_samples[:nums]
    else:
        all_samples = padding_data_helper(all_samples, nums)
    return target_samples, all_samples
