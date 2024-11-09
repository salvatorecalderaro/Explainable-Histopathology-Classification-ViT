import platform
import torch
import cpuinfo
import random 
import os 
import numpy as np
import argparse
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import timm
from time import perf_counter as pc
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from sklearn.metrics import accuracy_score,recall_score
import pandas as pd
import yaml

seed=0
nfolds=5
img_size=(224,224)
lr=0.002
epochs=30
mini_batch_size=128
nclasses=2
mean=np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])


def set_seed(seed):
    """
    This function sets the random seed for reproducibility across different libraries.

    Parameters:
    seed (int): The seed value to be used for setting the random seed.

    Returns:
    None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)



def identify_device():
    """
    This function identifies the device (CPU or GPU) and its name.

    Parameters:
    None

    Returns:
    A tuple containing the identified device and its name.
    """
    so = platform.system()
    if (so == "Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
            set_seed(seed)
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--type", type=str,required=True, choices=["tiny","small","base"], help="ViT architecture")
    args = parser.parse_args()
    t=args.type
    return t

def read_image(img_path):
    return np.array(Image.open(img_path).resize(img_size))
        
def load_data():
    images,labels=[],[]
    mapping_labels={
        "Abnormal":1,
        "Normal":0
    }
    
    path="../dataset"
    print("Uploading dataset.....")
    for folder in os.listdir(path):
        label=mapping_labels[folder]
        folder_path=os.path.join(path,folder)
        for imname in os.listdir(folder_path):
            img_path=os.path.join(folder_path,imname)
            image=read_image(img_path)
            images.append(image)
            labels.append(label)
    
    print("Number of patches in the dataset:",len(images))
    print("Number of classes",len(np.unique(labels)))

    images=np.array(images)
    labels=np.array(labels)
    
    return images,labels   
       

def create_train_test_sets(train,test,images,labels):
    x_train, x_test = images[train], images[test]
    y_train, y_test = labels[train], labels[test]
    
    
    data=list(zip(x_train,y_train))
    print(f"Number of images in the training set:",len(data))
    trainloader=DataLoader(data,shuffle=True,batch_size=mini_batch_size)
    
    data=list(zip(x_test,y_test))
    print(f"Number of images in the test set:",len(data))
    testloader=DataLoader(data,shuffle=False,batch_size=mini_batch_size)
    
    return trainloader, testloader

def load_vit(device,arch):
    name=f"vit_{arch}_patch16_224"
    model = timm.create_model(name, pretrained=False,num_classes=nclasses)
    model=model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    model=model.to(device)
    return model

def train_vit(device,model,trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2,gamma=0.7)
    model=model.float()
    start = pc()
    print("Training..")
    for epoch in tqdm(range(epochs)):
        model.train(True)
        running_loss = 0
        for i , data in enumerate(trainloader):
            inputs, labels = data
            inputs=inputs/255.0
            inputs = inputs.permute(0, 3, 1, 2)
            inputs =normalize(inputs, mean, std).to(device)     
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        running_loss /= (i + 1)
        #print('Epoch {}, loss {:.4f}'.format(epoch+1, running_loss))
        scheduler.step()
    
    t = pc()-start
    training_time = timedelta(seconds=t)
    print("Training ended in:", str(training_time))
    torch.cuda.empty_cache()
    return model,t

def save_model(model,arch):
    path=f"../experiments/{arch}/model.pt"
    torch.save(model.state_dict(), path)


def predict(device,model,testloader):
    y_true,y_pred=[],[]
    model.eval()
    start=pc()
    with torch.no_grad():
        for inputs,labels in testloader:
            inputs=inputs/255.0
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = normalize(inputs, mean, std).to(device)
            labels = labels.to(device)
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            y_true+=labels.tolist()
            y_pred+=preds.tolist()
    

    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    torch.cuda.empty_cache()
    return y_true,y_pred


def evaluate_model(f,target,predictions):
    acc=accuracy_score(target, predictions)
    sens=recall_score(target, predictions,pos_label=1)
    spec=recall_score(target,predictions,pos_label=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    
    return [f,acc,sens,spec]

def save_report(results,arch):
    
        
    columns = ["Fold", "Accuracy", "Sensitivity","Specificity","Training Time (s)"]
    path=f"../experiments/{arch}/metrics.csv"
    metrics = pd.DataFrame(results, columns=columns)
    metrics.to_csv(path, index=False)
    
    data = {}
    print(f"Average results:")
    for column in metrics.columns[1:]:
        values = metrics[column]
        mu, sigma = np.mean(values), np.std(values)
        print(f"{column}: mean {mu} sd {sigma}")
        data[column] = {
            "Mean": float(mu),
            "Standard Deviation": float(sigma)
        }

    path=f"../experiments/{arch}/results.yaml"
    with open(path, "w") as file:
        yaml.dump(data, file)

    
    
def run_exp(device,arch):
    images,labels=load_data()
    report=[]
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    
    for fold, (train_index, test_index) in enumerate(skf.split(images, labels),start=1):
        print(f"----------------------------------------------------")
        print(f"Fold {fold}")
        trainloader,testloader=create_train_test_sets(train_index, test_index,images,labels)
        model=load_vit(device,arch)
        model,t=train_vit(device,model,trainloader)
        target,predictions=predict(device,model,testloader)
        rep=evaluate_model(fold,target,predictions)
        rep.append(t)
        report.append(rep)
        if fold==1: save_model(model,arch)
        print(f"----------------------------------------------------")
    
    save_report(report,arch)    
    



def main():
    device,dev_name = identify_device()
    arch=parse_arguments()
    print(f"================================================")
    print(f"Using {device}  - {dev_name}")
    print(f"Classifying GasHisSDB images...")
    print(f"ViT - {arch}")
    print(f"================================================\n")
    
    run_exp(device,arch)
    exit(0)
    

if __name__ == "__main__":
    main()