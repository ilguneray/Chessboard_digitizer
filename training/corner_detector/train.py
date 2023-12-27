from load_data import LoadData
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..models.corner_model import Yolov1CornerDetector

@torch.no_grad()
def test_loss_calc(test_data_custom,
                   model,
                   device,
                   loss_fn):
    model.eval()
    with torch.no_grad():
        running_loss =0.0
        count=0
        for i,data in enumerate(test_data_custom,0):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            outputs=model(inputs)
            loss=loss_fn(outputs,labels)
            running_loss += loss.item()
            count+=1
    model.train()
    return running_loss/count

if __name__=="__main__":

    ##data_dir= your data path
    ## data_dir
    ## |____ train
    ## |____ test 
    ## data_dir folder should contain train and test folder in it.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data,test_data=LoadData(data_dir="",  #your data path goes here
                                  img_shape=(1200,800),
                                  model_input_shape=(448,448),
                                  batch_size=2,
                                  num_workers=2)
    model=Yolov1CornerDetector().to(device)
    loss_fn=nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay = 0.001, momentum = 0.9) 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,200,400,600,1000], gamma=0.6)

    train_loss_all=[]
    test_loss_all=[]
    epochs=2000
    best_loss=1e+6
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        count=0
        model.train()
        for i,data in tqdm(enumerate(train_data,0)):
            inputs,labels=data
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=loss_fn(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count+=1

        test_loss=test_loss_calc(test_data,model,device,loss_fn)
        print('[%d, %5d] TRAIN LOSS: %.3f' %(epoch , i , running_loss / count))
        print('[%d, %5d] TEST  LOSS: %.3f' %(epoch , i , test_loss))
        train_loss_all.append(running_loss / count)
        test_loss_all.append(test_loss) 

        if best_loss>running_loss / count:
            best_loss=running_loss / count
            torch.save(model.state_dict(),f"weights/{epoch}.pth")

        torch.save(model.state_dict(),f"weights/last.pth")
        print(f"BestLoss:\n",best_loss)
        scheduler.step()

        plt.plot(train_loss_all)
        plt.savefig("figures/train_loss.png")
        plt.close()
        plt.plot(test_loss_all)
        plt.savefig("figures/test_loss.png")
        plt.close()