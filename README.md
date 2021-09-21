# ExplainMLModels
Explain ML Models with streamlit app, using GradCam methods.

Demo:
https://explainpytorch.herokuapp.com/



## Upload Model
Models should be uploaded in a zip file. Zip file must contain a single main folder having a model name, and this folder must contain: 

* class_dict.pkl : Dictionary that contains index of prediction and corresponding class. If model predictions are in form of 0,1,2; class_dict.pkl should be ```{0: "Class A", 1:"Class B", 2:"Class C"}```

* transform_preprocess.py : py file that contains "transform_image" function.  This function should take image path, and transform image, then returns transformed image. An example form is :
```
from torchvision import transforms
from PIL import Image
import torch

    def transform_image(image_path):
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

return data_transforms(Image.open(image_path))
```
* model.pyt : model file that has .pyt fiel extension, saved by torch.save(model.state_dict(),"model.pyt") . You can edit the extension of saved Python models as .pyt, instead of .pkl, which will make it work. 

* model_script.py : model script that includes class,which strictly having name "Model", in such a format: 
```
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
```



