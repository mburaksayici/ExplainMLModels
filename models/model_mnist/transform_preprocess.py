from torchvision import transforms
from PIL import Image
import torch

def transform_image(image_path):
    data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    torch_img = data_transforms(Image.open(image_path))

    return torch.reshape(torch_img,(-1,1,28,28))

