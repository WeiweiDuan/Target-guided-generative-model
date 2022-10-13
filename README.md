# Target Guided Generative Model (TGGM)

## Paper link: https://arxiv.org/abs/2112.05786

## Paper citation:

@article{duanguided,<br/>
  title={Guided Generative Models using Weak Supervision for Detecting Object Spatial Arrangement in Overhead Images},  <br/>
  author={Duan, Weiwei and Chiang, Yao-Yi and Leyk, Stefan and Uhl, Johannes H and Knoblock, Craig A},  <br/>
  booktitle={2021 IEEE International Conference on Big Data (Big Data)},<br/>
  year={2021},  <br/>
  organization={IEEE}
}


## The goal is to detect the approximate locations of target objects within a region-of-interest using only one or a few manually labeled target object


### 1) Rerequired libraries
If cuda == 8.0, Python == 3.5, Keras == 2.1.6, tensorflow-gpu == 1.2.0 

If cuda == 10.0, Python == 3.6.9, Keras == 2.1.6, tensorflow == 1.7.0

Other packages: numpy, cv2, os, shutil, 

### Alternatively, all the required packages and libaries are in the docker imagery, which using cuda==10.0. 
**Here is the command to run the docker imagery**

<code>sudo nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888  pytorch/pytorch:1.2-cuda10.0-cudnn7-devel </code>

### 2) Run TGG
### Step 1: Data Generation
**Firstly, cropping images within the region level annotation**

The inputs are a map image and a mask for the region level annotation. 

The outputs are the cropped images saved in "./temp" folder

**** Here is the command to crop images
<code>python3 crop_images.py --data_dir {path-to-dataset} --loc_idx {name-or-index-of-image} --mask_name {region-annotation} --obj_name {target-object-name} --stride {sliding-window-stride} --win_size {sliding-window-size} </code>
  
**For example,** <code>python3 crop_images.py --data_dir './data/COWC/Toronto_03553.7.8' --loc_idx 'Toronto_03553.7.8' --mask_name None --obj_name='car' --stride=20 --win_size 50 </code>

**Secondly, choosing target images as labeled data from the cropped images**

You choose one or few target images in the ./temp folder, and put the chosen images' names in the "--pos_names" parameters. Please separated the names by commas

Besides the "--pos_names", generating training data needs parameters: dataset name (--dataset_name), image name (--loc_idx), target object name (--obj_name), and the proportion of cropped images using for training (--sample_portion)

The outputs are 1) labeled target images in a folder named as "positive" in a folder in the same directory of map image, 2) unlabeled cropped images in a folder named as "subset" in a folder in the same directory of map image. The number of images in the "subset" folder depends on the "--sample_portion".

**** Here is the command to generate training data
<code>python3 gen_train_data.py --dataset_name {dataset-name} --loc_idx {name-or-index-of-image} --obj_name {target-object-name} --pos_names {list-of-target-images-names} --sample_portion {percentage-of-cropped-images} </code>

**For example,** <code>python3 gen_train_data.py --dataset_name 'COWC' --loc_idx 'Toronto_03553.7.8' --obj_name='car' --pos_names '100_100' --sample_portion 1.0</code>

### Step 2: Iteratively Training TGG
**TGG takes cropped images and labeled target image(s) as inputs.<br/>

The outputs are predicted target images saved in a folder named by "--save_detected_images". <br/>

The images' names in the folder are the top-left coordinates of cropped images.**

<code>python3 train.py --map_path {path-to-map-path} --label_img_dir {dir-to-labeled-img} --image_size {sliding-window-size} --stride {sliding-window-stride} --num_epochs {number-of-epochs} --learning_rate {learning-rate} --weight {weight-for-CEloss} --saved_model_dir {dir-to-save-model} --saved_model_name {saved-model-name} --save_detected_images </code>
  
**For example,** <code>python3 train.py map_path='./data/COWC/Toronto_03553.7.8/Toronto_03553.7.8.jpg' --image_size 50 --stride 20 --num-epochs 500 --learning_rate 0.0001 --weight 500 --saved_model_path 'TGG.hd5f' </code>

### Step 3: Evaluation
**The evaluation script takes bounding-box level ground truth and image-level recognition results as inputs.<br/> 
The results are evaluated by precision, recall, and F1 score in the grid level.**

<code>python3 cal_eval.py --annotation_dir {path-to-ground-truth-folder} --pred_dir {path-to-TGG-ressult-folder} --obj_name {target-object-name} --loc_idx 'Toronto_03559.8.2' --grid_size {grid-size-for-eval} --image-size {crop-image-size}</code>
