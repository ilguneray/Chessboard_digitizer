from tqdm import tqdm
import json
import glob

## data_dir= your data path
## data_dir
## |____ train
## |____ test 
## data_dir folder should contain train and test folder in it.

data_dir=""   ### your data path goes here
train=data_dir+"train"
test=data_dir+"test"

def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

classes={"r":0,"n":1,"b":2,"k":3,"q":4,"p":5,"R":6,"N":7,"B":8,"K":9,"Q":10,"P":11}

def JsonToTxt(path):
    all_datas=glob.glob(path+"/*.json")
    for p in tqdm(all_datas):
        with open(p,"r") as f:
            data=json.load(f)["pieces"]

        for d in data:
            with open("d_txt/test/"+p[-9:-5]+".txt","a") as ff:
                c1,c2,c3,c4=coco_to_yolo(d["box"][0],d["box"][1],d["box"][2],d["box"][3],1200,800)
                ff.write(f"{classes[d['piece']]} {c1} {c2} {c3} {c4}\n")


if __name__=="__main__":
    data=JsonToTxt(test)













