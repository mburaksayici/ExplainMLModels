  
from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import numpy as np
import pandas as pd
from pandas import json_normalize
import urllib.request
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from PIL import Image
import requests
import os
import zipfile
import shutil

from explain_models import ModelLoader, Explainer

# Initialize parameters
# Define folder that contains model, loaded models will be extracted to models_folder_name .
models_folder_name = "models/"
explain_image_folder = "exp_images/"
explanation_methods = ["GradCam","XGradCam","EigenCam","AblationCam","ScoreCam","GradCam++"]



st.set_page_config(layout="wide")
sns.set_style('whitegrid')

# There'll be two pages, one for explaining  model, second for loading model.
my_page = st.sidebar.radio('Page Navigation', ['Explain Model', 'Load Model'])

if my_page == 'Explain Model':

    directories_in_models = os.listdir(models_folder_name) 
    # Zeroth Row
    page_1_row0_spacer1, page_1_row0_1, page_1_row0_spacer2, page_1_row0_2, page_1_row0_spacer3 = st.columns(
        (.1, 2, .2, 1, .1))
    page_1_row0_1.title('Explain Model')
    
    # First Row
    page_1_row1_spacer1, page_1_row1_1, page_1_row1_spacer2 = st.columns((.1, 3.2, .1))
    with page_1_row1_1:
        st.markdown("Explain PyTorch models with various methods.")

    # Second Row
    page_1_row2_spacer1, page_1_row2_1, page_1_row2_spacer2 = st.columns((.1, 3.2, .1))
    
    with page_1_row2_1:
        selected_model = st.selectbox("Select model", tuple(directories_in_models))
        selected_model += "/"
        
        # Load Model 
        model_loader = ModelLoader(models_folder_name+selected_model)
        model,class_dict,transform_func = model_loader.get_params()
        inv_class_dict =  {value:key for key, value in class_dict.items()}
        imgs_folder = models_folder_name+selected_model+"exp_images/"
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

    # Call explainer
    explain_model = Explainer(model,class_dict)
    explain_model.set_dataloader(transform_func)
    conv_layers = explain_model.get_conv_layers()
    conv_layers.reverse()
    # Obtain conv layers, in order to select at dropdown list
    conv_name_dict = dict(zip([str(i) for i in conv_layers],conv_layers))    
    
    
    
    page_1_row3_spacer1, page_1_row3_1, page_1_row3_spacer2 = st.columns((.1, 3.2, .1))
    with page_1_row3_1:
        #  Select Conv layer 
        selected_conv_layer_key = st.selectbox("Select Target CNN Layer, ordered from end to beginning of network", options =list(conv_name_dict.keys()))
        selected_conv_layer = conv_name_dict[selected_conv_layer_key]
        #  Select Explanation Method 
        selected_exp_method = st.selectbox("Select Explanation Method", explanation_methods)
        selected_class = st.selectbox("Select Class", list(class_dict.values()))
        #  Select Class to Explain  
        selected_class_index = int(inv_class_dict[selected_class])
        explain_model.set_target_layers(selected_conv_layer)
        
        
        
        mode_selection = st.radio('Load Image or Select from Disk', ["Load Image","Select Image"])

        if mode_selection == "Load Image":
            # Upload Image, save to disk
            uploaded_img = st.file_uploader('Upload Image', type=["jpg","jpeg","png","JPG","JPEG"])
            if (uploaded_img is not None):
                uploaded_img_path = models_folder_name+selected_model+explain_image_folder+uploaded_img.name
                with open(os.path.join(imgs_folder,uploaded_img.name),"wb") as f:
                    f.write(uploaded_img.getbuffer())
                    st.success("Saved File:{} to {}".format(uploaded_img.name,imgs_folder))
            explain_image = uploaded_img_path
        elif mode_selection == "Select Image":
            explain_images = os.listdir(models_folder_name+selected_model+explain_image_folder)
            selected_image = st.selectbox("Select Image", explain_images)
            explain_image = models_folder_name+selected_model+explain_image_folder+selected_image
        
         #selected_image
        # Set image path, transform
        explain_model.set_image_path(explain_image)
        explain_model.transform_image()

        # Explain image
        progress_bar = st.progress(0)
        if selected_exp_method == "XGradCam":
            cam_mask = explain_model.xgradcam_explainer(selected_class_index)
        elif selected_exp_method == "GradCam":
            cam_mask = explain_model.gradcam_explainer(selected_class_index)
        elif selected_exp_method == "EigenCam":
            cam_mask = explain_model.eigencam_explainer(selected_class_index)
        elif selected_exp_method == "AblationCam":
            progress_bar.progress(50) ## For now, it's a hack for slow grad-cam methods. It can be better represented with remaining time.
            cam_mask = explain_model.ablationcam_explainer(selected_class_index)
        elif selected_exp_method == "ScoreCam":
            progress_bar.progress(50) ## For now, it's a hack for slow grad-cam methods. It can be better represented with remaining time.
            cam_mask = explain_model.scorecam_explainer(selected_class_index)
        elif selected_exp_method == "GradCam++": 
            cam_mask = explain_model.gradcamplusplus_explainer(selected_class_index)
        progress_bar.progress(100)
        progress_bar.empty()
        # Get visualization
        vis = explain_model.visualize_with_mask(cam_mask)
        
        # Obtain prediction
        predicted_class = explain_model.predict()
        
        # Display prediction and image
        st.markdown("Model Prediction: {}".format(predicted_class))
        st.image([explain_model.raw_image.resize((vis.shape[0],vis.shape[1])),vis], caption=["Raw Image","Explanation"], width=vis.shape[0], use_column_width=None, clamp=False, channels='RGB', output_format='auto')



        
        
        
        
        
        
        
        
else:
    page_2_row0_spacer1, page_2_row0_1, page_2_row0_spacer2, page_2_row0_2, page_2_row0_spacer3 = st.columns(
        (.1, 2, .2, 1, .1))
    page_2_row0_1.title('Load Model')
    
    uploaded_zip = st.file_uploader('Model Zip File', type="zip")
    st.markdown("{}".format(uploaded_zip))
    if (uploaded_zip is not None):
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall("models/")


    st.markdown("""Models can be loaded in here, in the zip format. Zip must contain a folder, which includes for files:
    
    * class_dict.pkl : Dictionary that contains index of prediction and corresponding class. If model predictions are in form of 0,1,2; class_dict.pkl should be {0: "Class A", 1:"Class B", 2:"Class C"}

    * transform_preprocess.py : py file that contains "transform_image" function.  This function should take image path, and transform image, then returns transformed image. An example form is :
        
        def transform_image(image_path):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return data_transforms(Image.open(image_path))
    
    * model.pyt : model file that has .pyt fiel extension, saved by torch.save(model.state_dict(),"model.pyt") 
    
    * model_script.py : model script that includes class,which strictly having name "Model", in such a format: 
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    from torchvision import datasets, models, transforms

    class Model(nn.Module): ### It should strictly be "Model"
    def __init__(self):
        super(Model, self).__init__() ### It should strictly be "Model"
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

        

    
    
    """)

    
    

    
