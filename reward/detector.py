import tensorflow.compat.v1 as tf
import clip
import numpy as np
import cv2
import collections
from easydict import EasyDict
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time
from reward.vild import vild

class VILD():
    def __init__(self):
        # init clip model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip_model.cuda().eval()
        # init vild model
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
        saved_model_dir = "./reward/image_path_v2"
        _ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)
        self.session = session

        # vild setting
        #@markdown ViLD settings.
        max_boxes_to_draw = 12 #@param {type:"integer"}

        # Extra prompt engineering: swap A with B for every (A, B) in list.
        self.prompt_swaps = [('bowl', 'circle',)]

        nms_threshold = 0.3 #@param {type:"slider", min:0, max:0.9, step:0.05}
        min_rpn_score_thresh = 0.225  #@param {type:"slider", min:0, max:1, step:0.01}
        min_box_area = 200 #@param {type:"slider", min:0, max:10000, step:1.0}
        max_box_area = 4000  #@param {type:"slider", min:0, max:10000, step:1.0}
        self.vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area
    def vild_detect(self,image_path,category_names,verbose=False):
        category_name_string = ";".join(category_names)
        found_objects,image_withdetection = vild(self.session,
                                                       self.clip_model,
                                                       image_path, 
                                                       category_name_string, 
                                                       self.vild_params, 
                                                       plot_on=True, 
                                                       prompt_swaps=self.prompt_swaps,
                                                       verbose=verbose)
        return found_objects,image_withdetection