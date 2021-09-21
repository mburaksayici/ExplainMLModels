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
        """Creates model from model_script.py and returns the model.
        
        Returns:
            model_package_module.Model() : Model Class
        
        """
        model_package_file = (self.model_folder+ "model_script").replace("/",".")
        model_package_module = importlib.import_module(model_package_file)
        return model_package_module.Model()
    
    def load_model(self):
        """Load PyTorch model.
        
        Returns:
            model : PyTorch model with weights loaded
        
        """
        model = self.create_model()
        model_file = [i for i in self.files if "pyt" in i][0]
        model.load_state_dict(torch.load(self.model_folder+model_file,map_location=torch.device('cpu'))) ### In order to convert gpu models to cpu.
        model.eval() ## Needed in order to convert from training mode to eval mode.
        return model
    
    def load_class_dict(self):
        """Loads class dict from pickle.

        Returns:
            class_dict : dict Class dictionary {0:"Class A",1:"Class B"}

        """

        class_file = [i for i in self.files if "class_dict." in i][0]        
        with open(self.model_folder+class_file, 'rb') as pickle_file:
            class_dict = pickle.load(pickle_file)
        return class_dict
    
    def import_transform(self):
        """Imports transform function. Function should be located inside transform_preprocess.py that has a name of "transform_image".
        
        """
        
        transform_package_file = (self.model_folder+ "transform_preprocess").replace("/",".")
        transform_preprocess_module = importlib.import_module(transform_package_file)
        return transform_preprocess_module.transform_image
    
    def get_params(self):
        """Get parameters of model that's desired to be loaded.
        
        Returns:
            model : PyTorch model that weights are loaded.
            class_dict : Class dictionary
            transform_function : Raw image transform function.
        """
        
        model = self.load_model()
        class_dict = self.load_class_dict()
        transform_function = self.import_transform()
        return model,class_dict, transform_function
    
    
class Explainer:
    def __init__(self, model,class_dict):
        """Model is the PyTorch model,class_dict is the dictionary of class with keys are index and values are class names."""
        self.model = model
        self.class_dict = class_dict

        
    def set_dataloader(self,dataloader):
        """Set dataloader function that will remain unchanged and applied to all given images. Dataloader should take 
        image path and return transformed torch array which is manipulated with respect to same manner on testing.
        
        Args:
            dataloader: dataloader function
        """
        
        self.dataloader = dataloader
    
    def set_image_path(self,image_path):
        """Set the image path that will be explained

        Args:
            image_path: Image path in string format.
        """
        
        self.image_path = image_path
        
    def set_target_layers(self,target_layer):
        """Set the target layer extracted from the defined model.

        Args:
            target_layer: PyTorch layer
        """
        self.target_layers = target_layer
        
    def predict(self):
        """Predicts the transformed image to return predicted class of explained image.
        
        Returns:
            predicted_class : Name of the predicted class obtained by class_dict

        """
            
        prediction = self.model(self.transformed_image)
        predicted_class = self.class_dict[int(torch.argmax(prediction))]
        return predicted_class

    def get_children(self,model):
        """Obtains flattened the layers of sequential models of PyTorch. It's needed for architectures such as ResNet, that blocks contains more than one layers. This functions separates blocks. References: https://stackoverflow.com/a/65112132/6806531
        
        Args:
            model: PyTorch model
        
        Returns:
            flatt_children : List of separate layers

        """

        children = list(model.children())
        flatt_children = []
        if children == []:
            # if model has no children; model is last child! :O
            return model
        else:
           # look for children from children... to the last child!
           for child in children:
                try:
                    flatt_children.extend(self.get_children(child))
                except TypeError:
                    flatt_children.append(self.get_children(child))
        return flatt_children

    def get_conv_layers(self):
        """Filters convolutional layers on list of layers of PyTorch model. List of convolutional layers needed to choose which convolutional layer to explain, since Grad-Cam method works on convolutional layers.
        
        Returns:
            conv_layers : List of convolutional layers.

        """
        
        self.all_layers = self.get_children(self.model)
        conv_layers = [i for i in self.all_layers if  type(i) == torch.nn.modules.conv.Conv2d ]
        return conv_layers
        
    def transform_image(self):
        """Transforms image to prepare for explainer(PyTorch) models,keeps raw image at self.raw_image, transformed image at self.transformed_image, number of channels(to detect if image is grayscale/RGB) at self.channel.

        """

        self.raw_image = Image.open(self.image_path)
        self.transformed_image = self.dataloader(self.image_path)
        self.channels = self.transformed_image.size(1)
        
    def ablationcam_explainer(self,target_category):
        """AblationCam explainer. Explains why PyTorch model explains target_category by creating heatmap.
        
        Args:
            target_category: binary value that represents class.
        
        Returns:
            grayscale_cam : np.array Explanation mask created by AblationCam.

        """
        
        cam = AblationCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def eigencam_explainer(self,target_category):
        """EigenCam explainer. Explains why PyTorch model explains target_category by creating heatmap.
        
        Args:
            target_category: binary value that represents class.
        
        Returns:
            grayscale_cam : np.array Explanation mask created by AblationCam.

        """

        cam = EigenCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def scorecam_explainer(self,target_category):
        """ScoreCam explainer. Explains why PyTorch model explains target_category by creating heatmap.
        
        Args:
            target_category: binary value that represents class.
        
        Returns:
            grayscale_cam : np.array Explanation mask created by AblationCam.

        """
        
        cam = ScoreCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def gradcam_explainer(self,target_category):
        """GradCam explainer. Explains why PyTorch model explains target_category by creating heatmap.
        
        Args:
            target_category: binary value that represents class.
        
        Returns:
            grayscale_cam : np.array Explanation mask created by AblationCam.

        """
        
        cam = GradCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def gradcamplusplus_explainer(self,target_category):
        """GradCam++ explainer. Explains why PyTorch model explains target_category by creating heatmap.
        
        Args:
            target_category: binary value that represents class.
        
        Returns:
            grayscale_cam : np.array Explanation mask created by AblationCam.

        """
      
        cam = GradCAMPlusPlus(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def xgradcam_explainer(self,target_category):
        """XGradCam explainer. Explains why PyTorch model explains target_category by creating heatmap.
        
        Args:
            target_category: binary value that represents class.
        
        Returns:
            grayscale_cam : np.array Explanation mask created by AblationCam.

        """

        cam = XGradCAM(model=self.model, target_layer=self.target_layers)
        grayscale_cam = cam(input_tensor=self.transformed_image, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam

    def visualize_with_mask(self,grayscale_cam):
        """Combines raw image with mask. It's channel aware, when number of channel is 1, explained image is assumed to be grayscale, when number of channel is 3, it's assumed to be RGB.
        
        Args:
            grayscale_cam: Explanation mask that covers raw image.
        
        Returns:
            visualization : np.array Explained image on NumPy array.

        """

        final_img = np.array(self.raw_image.resize((grayscale_cam.shape[0],grayscale_cam.shape[1])))/255

        if len(np.array(self.raw_image).shape) ==2 or np.array(self.raw_image).shape[0] == 1 or np.array(self.raw_image).shape[-1] == 1 : 
            final_img = np.array([final_img,final_img,final_img])
            final_img = np.moveaxis(final_img,0,2)
        visualization = show_cam_on_image(final_img, grayscale_cam)
        return visualization
    
    
