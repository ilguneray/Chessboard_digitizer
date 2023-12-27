from ultralytics import YOLO

## This is ordinary yolov8 training.
## You can check how to do training in yolov8 on youtube etc.

if __name__=="__main__":
    yaml_dir=""#### your yaml file path goes here
    model = YOLO('yolov8x.pt')
    results = model.train(data=yaml_dir,epochs=100,imgsz=360,device=0,batch=8)

