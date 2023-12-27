from torch.utils.data import DataLoader,Dataset
import glob
import numpy as np
from PIL import Image
import json
from typing import Tuple,List
import torch
from torchvision import transforms

class LoadImagesFromFolder(Dataset):
    def __init__(self,target_dir:str,transforms,model_dimension:tuple,img_data_dimension:tuple):
        self.paths_im=np.array(glob.glob(target_dir+"/*.png"))
        self.paths_label=np.array(glob.glob(target_dir+"*/*.json"))
        self.transforms=transforms
        self.width_height_data=img_data_dimension
        self.width_height_model_input=model_dimension

    def load_image(self,index:int)->Image.Image:
        image_path=self.paths_im[index]
        return Image.open(image_path)

    def RemakePointsLocation(self,point:list):
        winput,hinput=self.width_height_data                
        wmodel,hmodel=self.width_height_model_input         
        divider=np.array([wmodel/winput,hmodel/hinput,wmodel/winput,hmodel/hinput,
                        wmodel/winput,hmodel/hinput,wmodel/winput,hmodel/hinput])

        point*=divider
        return point

    def load_label(self,index:int)->List:
        label_path=self.paths_label[index]
        with open(label_path,"r") as f:
            data=json.load(f)
        data=list(np.resize(data["corners"],(1,8)).squeeze().astype(np.float32))
        data=self.RemakePointsLocation(data)
        return data

    def __len__(self)->int:
        return len(self.paths_im)

    def __getitem__(self, index) -> Tuple[torch.Tensor,List]:
        img=self.load_image(index)
        label=self.load_label(index)
        if self.transforms:
            return self.transforms(img),torch.tensor(label)
        else:
            return img,label



def LoadData(data_dir,
             img_shape,
             model_input_shape,
             batch_size,
             num_workers,
             ):

    train_dir=data_dir+"train"
    test_dir=data_dir+"test"
    
    
    train_transforms=transforms.Compose([transforms.Resize(size=model_input_shape),
                                        transforms.ToTensor()])
    test_transforms=transforms.Compose([transforms.Resize(size=model_input_shape),
                                        transforms.ToTensor()])

    train_data=LoadImagesFromFolder(train_dir,transforms=train_transforms,
                                    model_dimension=model_input_shape,
                                    img_data_dimension=img_shape)
    test_data=LoadImagesFromFolder(test_dir,transforms=test_transforms,
                                    model_dimension=model_input_shape,
                                    img_data_dimension=img_shape)

    train_data_custom=DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True)

    test_data_custom=DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True)
    
    return train_data_custom,test_data_custom
    