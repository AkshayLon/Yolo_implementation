import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os, torch
import numpy as np

class ImageDataset(Dataset):

    def __init__(self):
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(current_dir, "exposure_5000_cam4")
        training_reference = dict()
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Resize((448,448))
        ])
        for f in os.listdir(data_dir):
            if f.endswith("png"):
                img = Image.open(os.path.join(data_dir, f))
                transformed_img = transform(img)
                training_reference[transformed_img] = f[:len(f)-4]
        self.x, self.y = list(), list()
        for x in training_reference:
            with open(file=os.path.join(data_dir, f"{training_reference[x]}.txt")) as f:
                data = f.read()
                split_boxes = data.split("\n")
                bounding_boxes = list(list(float(n) for n in box.split(" ")[1:]) for box in split_boxes if len(box)>0)
                if len(bounding_boxes)==1:
                    bounding_boxes.append([-1,-1,-1,-1])
                self.x.append(x)
                self.y.append(bounding_boxes)
        self.x = torch.from_numpy(np.array(self.x))
        self.y = torch.from_numpy(np.array(self.y))
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
        

        