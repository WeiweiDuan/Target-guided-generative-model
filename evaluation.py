import numpy as np
import cv2, os
from utils import helper
import xml.etree.ElementTree as ET

def parse_object_bbox(xml_file, obj_name):
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
            obj_bbox_list.append([ymin,xmin,ymax,xmax])
    return obj_bbox_list

def gen_recognition_mask(pred_path, img_map_shape, img_size):
    pred_points = np.loadtxt(pred_path, dtype='int32', delimiter=',')
    if len(pred_points.shape) == 1:
        pred_points = np.expand_dims(pred_points,axis=0)
    recog_mask = np.zeros(img_map_shape[:2])
    for i in pred_points:
        xmin, ymin = i[0]-img_size//2, i[1]-img_size//2
        xmax, ymax = i[0]+img_size//2, i[1]+img_size//2
        recog_mask[xmin:xmax,ymin:ymax] = 1
    return recog_mask

def gen_gt_mask(gt_path, target_obj_name, img_map_shape):
    if gt_path[-4:] == '.xml':
        gt_bbox = parse_object_bbox(gt_path, target_obj_name)
    else:
        gt_bbox = np.loadtxt(gt_path, dtype='int32', delimiter=',')

    gt_mask = np.zeros(img_map_shape[:2])
    for i in gt_bbox:
        xmin, ymin = i[0], i[1]
        xmax, ymax = i[2], i[3]
        gt_mask[xmin:xmax,ymin:ymax] = 1
    return gt_mask

def intersection2grid(mask, grid):#bbox, grid = [xmin,ymin,xmax,ymax]
    xmin, xmax = grid[0], grid[2]
    ymin, ymax = grid[1], grid[3]
    # compute the area of intersection rectangle
    inter_area = np.count_nonzero(mask[xmin:xmax+1, ymin:ymax+1])
    grid_area = (grid[0]-grid[2]) * (grid[1]-grid[3])
    return inter_area / grid_area*1.0

def intersection2bbox(bbox, grid): #bbox, grid = [xmin,ymin,xmax,ymax]
    xA = max(bbox[0], grid[0])
    yA = max(bbox[1], grid[1])
    xB = min(bbox[2], grid[2])
    yB = min(bbox[3], grid[3])
    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    bbox_area = (bbox[0]-bbox[2]) * (bbox[1]-bbox[3])
    if bbox_area <=0:
        return 0
    return inter_area / bbox_area*1.0

def gen_gt_grids(gt_mask, gt_path, target_obj_name, grid_size, overlapping_thres):
    true_pos_grids = []
    true_neg_grids = []
    if gt_path[-4:] == '.xml':
        gt_bbox = parse_object_bbox(gt_path, target_obj_name)
    else:
        gt_bbox =  np.loadtxt(gt_path, dtype='int32', delimiter=',')
    for row in range(0,gt_mask.shape[0],grid_size):
        for col in range(0,gt_mask.shape[1],grid_size):
            grid_xmin = row 
            grid_ymin = col 
            grid_xmax = row + int(grid_size)
            grid_ymax = col + int(grid_size) 
            grid_x_c, grid_y_c = row+grid_size//2, col+grid_size//2
            grid = [grid_xmin,grid_ymin,grid_xmax,grid_ymax]
            inter_grid = intersection2grid(gt_mask,grid)
            flag_inter_box = 0
            for bbox in gt_bbox:
                inter_bbox = intersection2bbox(bbox, grid)
                if inter_bbox >= overlapping_thres:
                    flag_inter_box = 1
                    break
            if inter_grid >= overlapping_thres or flag_inter_box == 1:
                true_pos_grids.append([grid_x_c, grid_y_c])
            else:
                true_neg_grids.append([grid_x_c, grid_y_c])
    return true_pos_grids, true_neg_grids

def gen_recog_grids(recog_mask, pred_points, grid_size, img_size, overlapping_thres):
    reg_pos_grids = []
    reg_neg_grids = []
    for row in range(0,recog_mask.shape[0],grid_size):
        for col in range(0,recog_mask.shape[1],grid_size):
            grid_xmin = row 
            grid_ymin = col 
            grid_xmax = row + int(grid_size)
            grid_ymax = col + int(grid_size) 
            grid_x_c, grid_y_c = row+grid_size//2, col+grid_size//2
            grid = [grid_xmin,grid_ymin,grid_xmax,grid_ymax]
            inter_grid = intersection2grid(recog_mask,grid)
            flag_inter_patch = 0
            for x, y in pred_points:
                patch_xmin, patch_xmax = x-img_size//2, x+img_size//2
                patch_ymin, patch_ymax = y-img_size//2, y+img_size//2
                patch = [patch_xmin, patch_ymin, patch_xmax, patch_ymax]
                inter_patch = intersection2bbox(patch, grid)
                if inter_patch >= overlapping_thres:
                    flag_inter_patch = 1
                    break
            if inter_grid >= overlapping_thres or flag_inter_patch == 1:
                reg_pos_grids.append([grid_x_c, grid_y_c])
            else:
                reg_neg_grids.append([grid_x_c, grid_y_c])
    return reg_pos_grids, reg_neg_grids

def precision_recall_f1(annotation_dir,pred_file,target_obj_name,\
                        loc_name,grid_size,img_size,img_map_shape,overlapping_thres):
    #pred_file = os.path.join(pred_dir, target_obj_name,\
    #                         "_".join([loc_name,target_obj_name,'pred_points.txt']))
    #pred_file = os.path.join(pred_dir, target_obj_name,\
    #                         "_".join([loc_name,target_obj_name,'VaDE','cluster1.txt']))
    #pred_file = os.path.join(annotation_dir,\
    #                         "_".join([loc_name,target_obj_name,'VaDE','cluster1.txt']))
    #pred_file = os.path.join("/data/weiweidu/Deep-Spectral-Clustering-using-Dual-Autoencoder-Network/res", "_".join([loc_name,target_obj_name,'cluster1.txt']))
    #pred_file = os.path.join("/data/weiweidu/Adversarial-Variational-Semi-supervised-Learning/res", "_".join(['xView',target_obj_name,loc_name,'pred_points.txt']))
    #pred_file = os.path.join(pred_dir,"_".join([loc_name,'pred_points.txt']))
         
    recog_mask = gen_recognition_mask(pred_file, img_map_shape, img_size)
    
    #gt_file = os.path.join(annotation_dir, loc_name+'.xml')
    #print(gt_file)
    #for COWC car
    #gt_file = os.path.join(annotation_dir, loc_name+'.gt_points.txt')
    #loc_name, loc_idx = loc_name.split('_')[0],loc_name.split('_')[1]
    #gt_file = os.path.join(annotation_dir, '_'.join([loc_name, target_obj_name,loc_idx,'gt','bbox.txt']))
    #for USGS wetland
    gt_file = os.path.join(annotation_dir, '_'.join([target_obj_name,'gt','bbox.txt']))
    gt_mask = gen_gt_mask(gt_file, target_obj_name, img_map_shape)
    
    true_pos_grids, true_neg_grids = gen_gt_grids(gt_mask, gt_file, target_obj_name, grid_size, overlapping_thres)
    
    pred_points = np.loadtxt(pred_file, dtype='int32', delimiter=',')
    if len(pred_points.shape) == 1:
        pred_points = np.expand_dims(pred_points,axis=0)
    reg_pos_grids, reg_neg_grids = gen_recog_grids(recog_mask, pred_points, grid_size, img_size, overlapping_thres)

    tp, fp, fn = 0.0, 0.0, 0.0
    for i in reg_pos_grids:
        flag = 0
        for j in true_pos_grids:
            if i == j:
                tp += 1
                flag = 1
                break
        if flag == 0:
            fp += 1

    for i in true_pos_grids:
        flag = 0
        for j in reg_pos_grids:
            if i == j:
                flag = 1
                break
        if flag == 0:   
            fn += 1
    if tp+fp==0 or tp+fn==0:
        return 0,0,0
    precision = tp / (tp+fp) *100.0
    recall = tp / (tp+fn)*100.0
    if (precision+recall) ==0 :
        return precision, recall, 0
    f1 = 2*precision*recall / (precision+recall)
    return precision, recall, f1
