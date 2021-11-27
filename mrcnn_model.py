#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 03:25:40 2021

@author: akhil
"""
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import numpy as np
import os

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    
    BACKBONE = 'resnet101'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.1

    STEPS_PER_EPOCH = 200
    
config = DetectorConfig()

class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
ROOT_DIR = './working'
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode='inference', 
                          config=inference_config, model_dir=ROOT_DIR)

weights_path=os.path.join(ROOT_DIR,"mrcnn_weights_resnet_101.h5")

# Load trained weights (fill in path to trained weights here)
assert weights_path != "", "Provide path to trained weights"
print("Loading weights from ", weights_path)
model.load_weights(weights_path, by_name=True)
# model.make_predict_function()
# model.keras_model._make_predict_function()


class MRCNN(object):   
    print("mrcnn loaded")
    
    class_names = ['pneumonia', 1, 'Lung Opacity']
    
    def get_colors_for_class_ids(self, class_ids):
        colors = []
        for class_id in class_ids:
            if class_id == 1:
                colors.append((.941, .204, .204))
        return colors
    
    def detect(self, image, fig): 
        min_conf = 0.96
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
            image, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    min_scale=config.IMAGE_MIN_SCALE,
                    max_dim=config.IMAGE_MAX_DIM,
                    mode=config.IMAGE_RESIZE_MODE)
            
            model.keras_model._make_predict_function()
            
            results = model.detect([image])
            r = results[0]

            assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
            if len(r['rois']) == 0:
                pass
            else:
                print(r['scores'])
                num_instances = len(r['rois'])

                for i in range(num_instances):
                    if r['scores'][i] > min_conf:
                        print("Pneumonia Detected")        
                
                        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        self.class_names, r['scores'], 
                                        colors=self.get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
            return r