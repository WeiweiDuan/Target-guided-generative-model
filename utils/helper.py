import os, cv2
import shutil
import numpy as np
import copy
import xml.etree.ElementTree as ET

def create_folder(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
        
def remove_files(path):
    for root, directory, files in os.walk(path):
        for fname in files:
            os.remove(os.path.join(root, fname))
    return 0

def load_file_names(dir_name):
    file_names = []
    for root, directory, files in os.walk(dir_name):
        for fname in files:
            file_names.append(fname[:-4])
    return file_names

def move_files(src, dst, keyword=None):
    for root, directory, files in os.walk(src):
        for fname in files:
            if keyword == None:
                shutil.move(os.path.join(root, fname), os.path.join(dst, fname))
            elif keyword in fname:
                shutil.move(os.path.join(root, fname), os.path.join(dst, fname))
    return 0
    
def visualization_grids(map_img, point_list, img_size, color=(0, 0, 255)):
    pred_img = np.zeros((map_img.shape))
    thickness= -1

    for i in point_list:
        y_c, x_c = int(i[1]), int(i[0])
        ymin, xmin = y_c-int(img_size/2), x_c-int(img_size/2)
        ymax, xmax = y_c+int(img_size/2), x_c+int(img_size/2)
        pred_img = cv2.rectangle(pred_img,(ymin,xmin),(ymax,xmax),color, thickness)

    pred_img = pred_img.astype('uint8')
    map_img = map_img.astype('uint8')

    # Generate result by blending both images (opacity of rectangle image is 0.25 = 25 %)
    out = cv2.addWeighted(map_img, 1.0, pred_img, 0.25, 1.0)
    return out



def plot_points_on_img(img, point_list=None):
    if point_list is None:
        return img
    res_img = copy.deepcopy(img)
    for i in point_list:
        x, y = i[1], i[0]
        radius = 2
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness=2
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 1 px
        res_img = cv2.circle(res_img, (x,y), radius, color, thickness)
    return res_img

def plot_bbox_on_img(img_path, bbox_list=None): #x,y is for numpy, which is y,x to draw bbox using opencv
    img = cv2.imread(img_path)
    if bbox_list == None:
        return img
    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x, y = int((xmin+xmax)/2), int((ymin+ymax)/2)
        radius = 2
        # Blue color in BGR
        color = (0, 255, 0)
        # Line thickness of 2 px
        thickness=2
        cv2.rectangle(img, (ymin,xmin), (ymax,xmax), color, thickness)
    return img


def plot_mask_on_map(map_img, pred_img):
    if len(pred_img.shape) == 2:
        pred_img_3channel = np.zeros(map_img.shape)
        pred_img_3channel[:,:,0] += pred_img*255
    else:
        pred_img_3channel = pred_img
    pred_img_3channel = pred_img_3channel.astype('uint8')
    map_img = map_img.astype('uint8')

    # Generate result by blending both images (opacity of rectangle image is 0.25 = 25 %)
    out = cv2.addWeighted(map_img, 1.0, pred_img_3channel, 0.25, 1.0)
    return out

def parse_xml_bbox(xml_file, obj_name):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    obj_bbox_list = []
    for group in root.findall('object'):
        title = group.find('name')
        if title.text == obj_name:
            bbox = group.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            obj_bbox_list.append([xmin,ymin,xmax,ymax])
    return obj_bbox_list
    
