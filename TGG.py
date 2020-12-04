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



def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def qz_graph(x, y, intermediate_dim=512,latent_dim=32, initializer = 'glorot_normal'):
    concat = Concatenate(axis=-1)([x, y])
    layer1 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(concat)
    layer2 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(layer1)
    z_mean = Dense(latent_dim,kernel_initializer = initializer)(layer2)
    z_var = Dense(latent_dim,kernel_initializer = initializer)(layer2)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
    return z_mean, z_var, z

def qy_graph(x, num_cls=2, initializer = 'glorot_normal'):
    layer1 = Dense(256, activation='relu',kernel_initializer = initializer)(x)#256. 64
    layer2 = Dense(128, activation='relu',kernel_initializer = initializer)(layer1)#128, 32
    qy_logit = Dense(num_cls,kernel_initializer = initializer)(layer2)
    qy = Activation(tf.nn.softmax)(qy_logit)
    return qy_logit, qy

def px_graph(z, intermediate_dim=512, original_dim=40*40*3, initializer = 'glorot_normal'):
    layer1 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(z)
    layer2 = Dense(intermediate_dim, activation='relu',kernel_initializer = initializer)(layer1)
    reconstruction = Dense(original_dim, activation='sigmoid',kernel_initializer = initializer)(layer2)
    return reconstruction

def pzy_graph(y, latent_dim=32, initializer = 'glorot_normal'):
    h = Dense(16, activation='relu',kernel_initializer = initializer)(y)#128
    h = Dense(8, activation='relu',kernel_initializer = initializer)(h)#256, 64
    zp_mean = Dense(latent_dim,kernel_initializer = initializer)(h)
    zp_var = Dense(latent_dim,kernel_initializer = initializer)(h)
    return zp_mean, zp_var

def loss(x, xp, zm, zv, zm_prior, zv_prior, w_mse, w_kl):
    reconstruction_loss = mse(x, xp)
    reconstruction_loss *= w_mse
    kl_loss = (zv_prior-zv)*0.5 + (K.square(zm-zm_prior) + K.exp(zv)) / (2*K.exp(zv_prior)+1e-10) - 0.5
    kl_loss = K.sum(kl_loss, axis=-1) * w_kl
    return reconstruction_loss + kl_loss

def kl_loss(zm, zv, zm_prior, zv_prior, weight):
    loss = (zv_prior-zv)*0.5 + (np.square(zm-zm_prior) + np.exp(zv)) / 2*np.exp(zv_prior) - 0.5
    loss = np.sum(loss, axis=-1) * weight
    return loss

def mse_loss(x, xp, weight):
    return (np.square(x - xp)).mean(axis=None) * weight

def ce_loss(yp, weight):
    return (yp * np.log(yp / np.array([0.50,0.50]))+1e-10).mean(axis=None) * weight

def initialize_tgg(original_dim, num_cls, latent_dim, intermediate_dim, w_recons, w_kl, w_ce):
    x, y = Input(shape=(original_dim,)), Input(shape=(num_cls,))
    sub_enc = Model([x,y],qz_graph(x, y, intermediate_dim=intermediate_dim, latent_dim=latent_dim))
    z = Input(shape=(latent_dim,))
    sub_dec = Model(z, px_graph(z, intermediate_dim=intermediate_dim, original_dim=original_dim))

    x_u = Input(shape=(original_dim,), name='x_u')
    x_l = Input(shape=(original_dim,), name='x_l')
    y0 = Input(shape=(num_cls,), name='y0_inputs')
    y1 = Input(shape=(num_cls,), name='y1_inputs')

    pzy0 = Model(y0, pzy_graph(y0, latent_dim=latent_dim))
    pzy1 = Model(y1, pzy_graph(y1, latent_dim=latent_dim))
    zm_p0,zv_p0 = pzy0(y0)
    zm_p1,zv_p1 = pzy1(y1)

    # zm_p0,zv_p0 = pzy_graph(y0, latent_dim=latent_dim)
    # zm_p1,zv_p1 = pzy_graph(y1, latent_dim=latent_dim)


    # zm0, zv0, z0 = qz_graph(x_u, y0, intermediate_dim=intermediate_dim, latent_dim=latent_dim)
    zm0, zv0, z0 = sub_enc([x_u, y0])
    zm_l, zv_l, z_l = sub_enc([x_l, y0])
    zm1, zv1, z1 = qz_graph(x_u, y1, intermediate_dim=intermediate_dim, latent_dim=latent_dim)


    # xp_u0 = px_graph(z0, intermediate_dim=intermediate_dim, original_dim=original_dim)
    xp_u0 = sub_dec(z0)
    xp_l = sub_dec(z_l)
    xp_u1 = px_graph(z1, intermediate_dim=intermediate_dim, original_dim=original_dim)

    qy_logit, qy = qy_graph(x_u, num_cls=2)

    tgg = Model([x_u,x_l,y0,y1], [xp_l,xp_u0,xp_u1,qy,zm_l,zv_l,zm0,zv0,zm1,zv1,zm_p0,zv_p0,zm_p1,zv_p1])


    cat_loss = qy * K.log(qy / K.constant(np.array([0.5,0.5])))
    cat_loss = K.sum(cat_loss, axis=-1) * w_ce

    tgg_loss = qy[:,0]*loss(x_u,xp_u0,zm0,zv0,zm_p0,zv_p0,w_recons,w_kl)+\
                qy[:,1]*loss(x_u,xp_u1,zm1,zv1,zm_p1,zv_p1,w_recons,w_kl)+\
                loss(x_l,xp_l,zm_l,zv_l,zm_p0,zv_p0,w_recons,w_kl) + cat_loss

    tgg.add_loss(tgg_loss)
    return tgg

def train_tgg(model, optimizer, _x_u, _x_l_aug, epochs, batch_size, save_model_path, load_model_path=None):
    model.compile(optimizer=optimizer, loss=None)
    if load_model_path != None:
        model.load_weights(load_model_path)
    num_samples = _x_u.shape[0]
    model.fit([_x_u,_x_l_aug,np.array([[1,0]]*num_samples),np.array([[0,1]]*num_samples)],\
        epochs=epochs, batch_size=batch_size, verbose=1)
    model.save_weights(save_model_path)
