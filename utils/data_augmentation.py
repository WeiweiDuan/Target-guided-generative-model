import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def data_aug(samples, translation_list=[], rotation_angle=[]):
    img_aug = []
    for img in samples:
        if len(translation_list)!=0 and len(rotation_angle) != 0:
            trans_aug = []
            trans_aug.extend(translation(img, translation_list))
            for i in trans_aug:
                img_aug.extend(rotation(i, rotation_angle))
            img_aug.extend(trans_aug)
        elif len(translation_list) != 0:
            img_aug.extend(translation(img, translation_list))
        elif len(rotation_angle) != 0:
            img_aug.extend(rotation(img, rotation_angle))
        elif len(translation_list)==0 and len(rotation_angle) == 0:
            img_aug.append(img)
    img_aug = np.array(img_aug)
    #labels = np.array([[0]]*img_aug.shape[0]) 
    return img_aug
                             
def translation(img, shifting_list):
    rows,cols,channels = img.shape
    img_aug = []
    for h in shifting_list:
        for v in shifting_list:
            M = np.float32([[1,0,h],[0,1,v]])
            dst = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.INTER_AREA)
            #dst = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.cv2.BORDER_CONSTANT, borderValue=(0.49,0.51,0.58))
            img_aug.append(dst)
    return img_aug

#def rotate_image(mat, angle):
    #height, width = mat.shape[:2]
    #half_h, half_w = int(height/2), int(width/2)
    #image_center = (width / 2, height / 2)

    #rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    #radians = math.radians(angle)
    #sin = math.sin(radians)
    #cos = math.cos(radians)
    #bound_w = int((height * abs(sin)) + (width * abs(cos)))
    #bound_h = int((height * abs(cos)) + (width * abs(sin)))

    #rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    #rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    #rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    #c_x, c_y = int(rotated_mat.shape[0]/2), int(rotated_mat.shape[1]/2)
    #rotated_mat_crop = rotated_mat[c_x-half_h:c_x+half_h,c_y-half_w:c_y+half_w,:]
    # rotation calculates the cos and sin, taking absolutes of those.
    #abs_cos = abs(rotation_mat[0,0]) 
    #abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    #bound_w = int(height * abs_sin + width * abs_cos)
    #bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    #rotation_mat[0, 2] += bound_w/2 - image_center[0]
    #rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    #rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    #return rotated_mat 

def rotate_image(image, angle):
    image_height = image.shape[0]
    image_width = image.shape[1]
    diagonal_square = (image_width*image_width) + (
        image_height* image_height
    )
    #
    diagonal = int(math.sqrt(diagonal_square))
    padding_top = int((diagonal-image_height) / 2)
    padding_bottom = int((diagonal-image_height) / 2)
    padding_right = int((diagonal-image_width) / 2)
    padding_left = int((diagonal-image_width) / 2)
    padded_image = cv2.copyMakeBorder(image,
                                      top=padding_top,
                                      bottom=padding_bottom,
                                      left=padding_left,
                                      right=padding_right,
                                      borderType=cv2.INTER_AREA
                                      #borderType=cv2.BORDER_CONSTANT,
                                      #value=(0.49,0.51,0.58)
            )
    padded_height = padded_image.shape[0]
    padded_width = padded_image.shape[1]
    transform_matrix = cv2.getRotationMatrix2D(
                (padded_height/2,
                 padded_width/2), # center
                angle, # angle
      1.0) # scale
    rotated_image = cv2.warpAffine(padded_image,
                                   transform_matrix,
                                   (diagonal, diagonal),
                                   flags=cv2.INTER_LANCZOS4)
 
    c_x, c_y = int(padded_height/2), int(padded_width/2)
    c_h, c_w = int(image_height/2), int(image_width/2)
    rotated_image_crop = rotated_image[c_x-c_h:c_x+c_h, c_y-c_w:c_y+c_w, :]
    return rotated_image_crop

def rotation(img, angle_list):
    img_rotation = []
    for angle in angle_list:
        img_rotation.append(rotate_image(img, angle))
    return img_rotation
