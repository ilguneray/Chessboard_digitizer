import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.corner_model import Yolov1CornerDetector
import glob
import numpy as np
from ultralytics import YOLO
from utils.helpers import Helpers

@torch.no_grad()
def GetCorners(model:nn.Module,img_path:str,model_input_shape:list,transforms=None,device="cpu")->None:
    """
    Executes the model with given input image and returns the corner coordinates of chessboard contour.
    
    Arguments:
    model:nn.Module
    img_path:str
    model_input_shape:list
    transforms=torchvision.transforms |default=None
    device:str |default="cpu"

    Returns:
    np.ndarray
    """
    model.eval()
    img_types=["/*.png","/*.jpeg","/*.jpg"]
    all_images=[path for img_typ in img_types for path in glob.glob(img_path+img_typ)]
    model_input_shape=model_input_shape[0]
    if len(all_images)>0:
        for p in tqdm(all_images):
            img=Image.open(p).convert('RGB')
            test_img=transforms(img)
            output=model(torch.unsqueeze(test_img,0).to(device))
            w,h=img.size[0]/model_input_shape,img.size[1]/model_input_shape

            p1=int(round((output.cpu().numpy().squeeze()[0]*w))),int(round((output.cpu().numpy().squeeze()[1]*h)))
            p2=int(round((output.cpu().numpy().squeeze()[2]*w))),int(round((output.cpu().numpy().squeeze()[3]*h)))
            p3=int(round((output.cpu().numpy().squeeze()[4]*w))),int(round((output.cpu().numpy().squeeze()[5]*h)))
            p4=int(round((output.cpu().numpy().squeeze()[6]*w))),int(round((output.cpu().numpy().squeeze()[7]*h)))
            
    else:
        img=Image.open(img_path).convert('RGB')
        test_img=transforms(img)
        output=model(torch.unsqueeze(test_img,0).to(device))
        w,h=img.size[0]/model_input_shape,img.size[1]/model_input_shape

        p1=int(round((output.cpu().numpy().squeeze()[0]*w))),int(round((output.cpu().numpy().squeeze()[1]*h)))
        p2=int(round((output.cpu().numpy().squeeze()[2]*w))),int(round((output.cpu().numpy().squeeze()[3]*h)))
        p3=int(round((output.cpu().numpy().squeeze()[4]*w))),int(round((output.cpu().numpy().squeeze()[5]*h)))
        p4=int(round((output.cpu().numpy().squeeze()[6]*w))),int(round((output.cpu().numpy().squeeze()[7]*h)))
        
    return np.array([p1,p2,p3,p4])


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_input_shape=448,448           #w,h
    test_transforms=transforms.Compose([transforms.Resize(size=model_input_shape),
                                        transforms.ToTensor()])

    test_dir="test_images/1.png"
    model=YOLO("weights/piece_detection.pt")
    results=model.predict(test_dir)
    for result in results:
        boxes = result.boxes.xyxy.cpu()
        classes = [result.names[c.item()] for c in result.boxes.cls]

    corner_model=Yolov1CornerDetector().to(device)
    corner_model.load_state_dict(torch.load("weights/board_corners.pth",map_location=torch.device(device)))
    corners=GetCorners(corner_model,test_dir,model_input_shape,test_transforms,device)
    corners=Helpers.RelocateCorners(corners)

    im=cv2.imread(test_dir)
    actual_corners=Helpers.GetChessboardSquareCorners(corners)
    square_centers=Helpers.FindSquareCenters(actual_corners)
    chessboard=Helpers.FindCorrespondingSquare(square_centers,boxes,classes)
    fen=Helpers.ChessBoardSaveAsPNG(chessboard)

    for corner in actual_corners:
        cv2.circle(im,corner,4,(255,0,0),-1)
    for corner in square_centers:
        cv2.circle(im,corner,4,(0,255,255),-1)

    cv2.imwrite("outputs/square_centers_and_corners.png",im)
