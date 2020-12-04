from keras.layers import Lambda, Input, Dense, Merge, Concatenate,Multiply, Add, add, Activation
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import metrics

from utils import load_data, data_augmentation, helper
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from keras import metrics
import tensorflow as tf
import keras
import TGG

def process_inputs(_x_u, _x_l, batch_size):
    np.random.shuffle(_x_u)
    num_samples = _x_u.shape[0]//batch_size * batch_size
    print('num samples: ',  num_samples)
    print('unlabeled samples shape: ', _x_u.shape)
    print('labeled samples shape: ', _x_l.shape)
    _x_l_aug = data_augmentation.data_aug(_x_l, SHIFT_LIST, ROTATION_ANGLE)
    np.random.shuffle(_x_l_aug)
    print('augmented labeled samples shape: ', _x_l_aug.shape)

    _x_l = np.reshape(_x_l, [-1, IMG_SIZE*IMG_SIZE*3])
    _x_l_aug = np.reshape(_x_l_aug, [-1, IMG_SIZE*IMG_SIZE*3])
    _x_u = np.reshape(_x_u, [-1, IMG_SIZE*IMG_SIZE*3])

    _x_u = _x_u.astype('float32') / 255
    _x_l = _x_l.astype('float32') / 255
    _x_l_aug = _x_l_aug.astype('float32') / 255


    while _x_l_aug.shape[0] < _x_u.shape[0]:
        _x_l_aug = np.vstack((_x_l_aug, _x_l_aug[:_x_u.shape[0]-_x_l_aug.shape[0]]))

    np.random.shuffle(_x_l_aug)
    np.random.shuffle(_x_u)
    _x_l_aug = _x_l_aug[:num_samples]
    _x_u = _x_u[:num_samples]
    print('target samples shape: ', _x_l_aug.shape)
    print('all samples shape: ', _x_u.shape)
    return _x_u, _x_l_aug
    
def inference(model, x_test, x_test_idx, x_l_test, loc_name, thres, pos_folder_path, img_size=30):
    print('processing {:s}'.format(loc_name))
    while x_l_test.shape[0] < x_test.shape[0]:
        x_l_test = np.vstack((x_l_test, x_l_test[:x_test.shape[0]-x_l_test.shape[0]])) 

    x_l_test = x_l_test[:x_test.shape[0]]

    x_u_pred0,x_u_pred1,x_l_pred,y_u_pred,_,_,_,_,_,_,_,_,_,_ = model.predict([x_test,x_l_test,np.array([[1,0]]*x_test.shape[0])\
               ,np.array([[0,1]]*x_test.shape[0])])

    detected_idx = helper.load_file_names(pos_folder_path)
    print('num of detected: ', len(detected_idx))
    pred_pos_idx = []
    num_detected = len(helper.load_file_names(os.path.join('multi_check_data',loc_name)))
    while num_detected == 0 and thres>0.5:   
        for i in range(x_test.shape[0]): 
            if y_u_pred[i, 0] > thres:
                pred_pos_idx.append(x_test_idx[i])
                temp_name = loc_name + '_' + x_test_idx[i]
                if temp_name not in detected_idx:
                    print(x_test_idx[i],y_u_pred[i])
                    cv2.imwrite(os.path.join('multi_check_data',loc_name,'_'.join([loc_name,x_test_idx[i]+'.png'])), x_test[i].reshape((img_size, img_size, 3))*255)
        thres = thres - 0.0019
        num_detected = len(helper.load_file_names(os.path.join('multi_check_data',loc_name)))
        print(thres, num_detected)
    print('number of detected in {:s} is {:d}'.format(loc_name, num_detected))

#######################################################    
# hyperparamters
os.environ["CUDA_VISIBLE_DEVICES"]="2"
obj_name = 'ship'
DATA_DIR = '_'.join([obj_name, 'DIOR', 'test'])

SHIFT_LIST = []#[-2,-1,0,1,2] 
ROTATION_ANGLE = [180,360]#[90,180,270,360]#[45,90,135,180,225,270,315,360]#
IMG_SIZE = 30
STRIDE = 10
EPOCHS = 400
LEARNING_RATE = 0.0001
BATCH_SIZE = 300

latent_dim = 16#32, 5
intermediate_dim = 256#256 for dior ships
num_cls = 2
optimizer = Adam(lr=LEARNING_RATE)
# optimizer = RMSprop(lr=LEARNING_RATE)
initializer = 'glorot_normal'#'random_uniform'#
original_dim = IMG_SIZE*IMG_SIZE*3
w_recons, w_kl, w_ce = IMG_SIZE*IMG_SIZE*3.0, 1.0, 500.0
threshold = 0.5

path = os.path.join('multi_check_data',loc_idx)
pos_samples_path = os.path.join('multi_maps_data',obj_name,loc_idx,'pos_samples')
map_path = os.path.join(DATA_DIR, loc_idx+'.jpg')
SAVE_MODEL_PATH = os.path.join(obj_name,loc_idx,'_'.join(['TGG',loc_idx+'.hdf5']))
print(pos_samples_path)
print(map_path)
#######################################################

#######################################################
# Data Processing
labele_samples, _ = load_data.load_wetland_samples(pos_samples_path)
all_samples, _ = load_data.load_all_data(map_path, '', IMG_SIZE, STRIDE)
_x_u, _x_l_aug = process_inputs(all_samples, labele_samples, BATCH_SIZE)
print('processing {:d}, {:s}: '.format(i, loc_idx))
print('x_u, x_l_aug shape: ', _x_u.shape, _x_l_aug.shape)
#######################################################

#######################################################
# Training Phase
vae = TGG.initialize_tgg(original_dim, num_cls, latent_dim, intermediate_dim,w_recons, w_kl, w_ce)
optimizer = Adam(lr=LEARNING_RATE)
TGG.train_tgg(vae, optimizer, _x_u, _x_l_aug, EPOCHS, BATCH_SIZE, SAVE_MODEL_PATH, LOAD_MODEL_PATH)
#######################################################

#######################################################
# Testing Phase
vae = TGG.initialize_tgg(original_dim, num_cls, latent_dim, intermediate_dim,w_recons, w_kl, w_ce)
vae.load_weights(SAVE_MODEL_PATH)
pos_samples_path = os.path.join('multi_maps_data',obj_name,loc_idx,'pos_samples')
path = os.path.join('multi_check_data',loc_idx)
helper.create_folder(path)
helper.remove_files(path)
map_path = os.path.join(DATA_DIR, loc_idx+'.jpg')
x_test, x_test_idx = load_data.load_all_data(map_path, '', IMG_SIZE, STRIDE)
x_test = np.array(x_test) / 255.0
x_test = np.reshape(x_test, (-1, IMG_SIZE*IMG_SIZE*3))
threshold = 0.501
inference(vae, x_test,  x_test_idx, _x_l_aug, loc_idx, threshold, pos_samples_path)
#######################################################
