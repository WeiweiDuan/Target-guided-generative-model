# Target Guided Generative Clustering Model (TGG)
## The goal is to detect the approximate locations of target objects within a region-of-interest using only one or a few manually labeled target object

### 1) Rerequired libraries
Python == 3.5

Keras == 2.1.6 

tensorflow-gpu == 1.2.0 

cuda == 8.0 

numpy, cv2, os

### Alternatively, all the required packages and libaries are in the docker imagery. 
**Here is the command to run the docker imagery**

<code>sudo nvidia-docker run -t -i -v {local_dir}:{docker_dir} -p 8888:8888  spatialcomputing/map_text_recognition_gpu </code>

### 2) Run TGG
### Step 1: Data Generation
<mark> python3 crop_images.py --data_dir path-to-dataset --loc_idx name-or-index-of-image --mask_name region-annotation --obj_name target-object-name --stride sliding-window-stride --win_size sliding-window-size </mark>
  
For example, <mark>python3 crop_images.py --data_dir '/data/weiweidu/COWC/DetectionPatches_800x800/Toronto_ISPRS' --loc_idx 'Toronto_03559.8.2' --mask_name None --obj_name='car' --stride=20 --win_size 50 </mark>

gen_train_data.py

### Step 2: Iteratively Training TGG
train.py

### Step 3: Evaluation
