# Target Guided Generative Clustering Model (TGG)
## The goal is to detect the approximate locations of target objects within a region-of-interest using only one or a few manually labeled target object

### 1) Rerequired libraries
Python == 3.5

Keras == 2.1.6 

tensorflow-gpu == 1.2.0 

cuda == 8.0 

numpy, cv2, os


### 2) Run scripts
### Step 1: Data Generation
crop_images.py
gen_train_data.py

### Step 2: Iteratively Training TGG
train.py
### Step 3: Evaluation
