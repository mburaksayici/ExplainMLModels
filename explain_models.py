from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import importlib
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image




class ModelLoader:
    def __init__(self,model_folder):
        self.model_folder = model_folder
        self.files = os.listdir(model_folder)    
        
    def create_model(self):
        """Creates model from model_script.py and returns the model."""
        model_package_file = (self.model_folder+ "model_script").replace("/",".")
        model_package_module = importlib.import_module(model_package_file)
        return model_package_module.Model()
    
    def load_model(self):
        
        model = self.create_model()
        model_file = [i for i in self.files if "pyt" in i][0]
        model.load_state_dict(torch.load(self.model_folder+model_file))
        return model
    
    def load_class_dict(self):
        class_file = [i for i in self.files if "class_dict." in i][0]
        print(self.model_folder+class_file)
        
        with open(self.model_folder+class_file, 'rb') as pickle_file:
            class_dict = pickle.load(pickle_file)
        
        return class_dict
    def import_transform(self):
        transform_package_file = (self.model_folder+ "transform_preprocess").replace("/",".")
        print(transform_package_file)
        transform_preprocess_module = importlib.import_module(transform_package_file)
        return transform_preprocess_module.transform_image
    def get_params(self):
        
        model = self.load_model()
        class_dict = self.load_class_dict()
        transform_function = self.import_transform()
        return model,class_dict, transform_function
    
    
class Explainer:
    def __init__(self, model,class_dict):
        """Model is the PyTorch model,class_dict is the dictionary of class with keys are index and values are class names."""
        self.model = model
        self.class_dict = class_dict
        self.target_layers = self.model.layer4[-1]
        
    def set_dataloader(self,dataloader):
        """Set dataloader function that will remain unchanged and applied to all given images. Dataloader should take 
        image as a NumPy array and returns transformed array which is manipulated with respect to same manner on training.
        """
        self.dataloader = dataloader
    
    def set_image_path(self,image_path):
        self.image_path = image_path
        
    def transform_image(self):
        self.raw_image = Image.open(self.image_path)
        self.transformed_image = self.dataloader(self.image_path)
        if len(self.transformed_image.shape) ==3 : 
            self.transformed_image = self.transformed_image.unsqueeze_(0)

    def ablationcam_explainer(self,target_category):
        cam = AblationCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def gradcam_explainer(self,target_category):
        cam = GradCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def gradcamplusplus_explainer(self,target_category):
        cam = GradCAMPlusPlus(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def xgradcam_explainer(self,target_category):
        cam = XGradCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def scorecam_explainer(self,target_category):
        cam = ScoreCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def eigencam_explainer(self,target_category):
        cam = EigenCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def visualize_with_mask(self,grayscale_cam):
        visualization = show_cam_on_image(np.array(self.raw_image.resize((grayscale_cam.shape[0],grayscale_cam.shape[1])))/255, grayscale_cam)
        #Image.fromarray(visualization,"L").show()
        return visualization
    
    
