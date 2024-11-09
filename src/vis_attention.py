import random
import os 
import numpy as np
import torch
import cpuinfo
import platform
import argparse
from PIL import Image
import timm
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms.functional import normalize
from scipy.ndimage import gaussian_filter, median_filter
from skimage import morphology
import torch.nn.functional as F
import matplotlib.pyplot as plt



plt.rcParams["text.usetex"]=True

mean=np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])

seed=0
nfolds=5
nclasses=2
patch_size=16
image_size=224
image_size_patch=image_size//patch_size
alpha=0.8


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def identify_device(seed):
    """
    This function identifies the device (CPU or GPU) to be used for the computations.

    Parameters:
    None

    Returns:
    device (torch.device): The identified device for computations.

    The function first checks the operating system. If it's macOS, it checks if the MPS (Metal Performance Shaders) is available. If it is, it sets the device to MPS; otherwise, it sets it to CPU. If the operating system is not macOS, it checks if the CUDA (Compute Unified Device Architecture) is available. If it is, it sets the device to CUDA; otherwise, it sets it to CPU.

    After identifying the device, the function prints a message indicating the device and its name.
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
    return device,dev_name

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--type", type=str,required=True, choices=["tiny","small","base"], help="ViT architecture")
    args = parser.parse_args()
    t=args.type
    return t

def read_image(img_path):
    return np.array(Image.open(img_path).resize((image_size,image_size)))
        
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

def load_vit(arch,device):
    path=f"../experiments/{arch}/model.pt"
    name=f"vit_{arch}_patch16_224"
    model=timm.create_model(name,pretrained=True,num_classes=nclasses)
    model.load_state_dict(torch.load(path,weights_only=True))
    model=model.to(device)
    return model

    
def forward_wrapper(attn_obj):
    def forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 1:]
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return forward
    


def save_image(img,attention,target,predicition,i,arch,):
    path=f"../experiments/{arch}/attention maps/image_{i}.png"
    title="True Label: %s - Predicted Label: %s" %(target,predicition)
    fig, axs = plt.subplots(1, 3,constrained_layout=True)
    fig.suptitle(title,fontsize=20,y=0.85)
    fig.set_size_inches(18.5, 10.5)
    axs[0].imshow(img)
    axs[0].axis("off")

    axs[1].imshow(attention, cmap='viridis')
    axs[1].axis('off')

    axs[2].imshow(img)
    axs[2].imshow(attention, alpha=alpha, cmap='viridis', vmin=0)
    axs[2].axis("off")
    fig.savefig(path, bbox_inches='tight')
    plt.clf()
    plt.close()

def compute_att_maps(device,arch):
    images,labels=load_data()
    model=load_vit(arch,device)
    
    mapping={
        1:"Abnormal",
        0:"Normal"
    }
    
    skf=StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=seed)
    
    folds=skf.split(images,labels)
    
    _, test_index = next(folds)

    indicies=np.random.choice(test_index,50,replace=False)
    test_images,test_labels=images[indicies],labels[indicies]
    
    i=1
    model.eval()
    model.blocks[-1].attn.forward = forward_wrapper(model.blocks[-1].attn)
    with torch.no_grad():
        for (img,label) in zip(test_images,test_labels):
            
            img_tensor=torch.from_numpy(img).permute(2,0,1)
            img_norm=normalize(img_tensor/255,mean,std)
            outputs=model(img_norm.unsqueeze(0).float().to(device))
            _,pred=torch.max(outputs, 1)
            cls_weight = model.blocks[-1].attn.cls_attn_map
            cls_weight = model.blocks[-1].attn.cls_attn_map.mean(dim=1)
            cls_weight = cls_weight.view(image_size_patch,image_size_patch).detach()
            cls_resized = cls_weight.view(1, 1, image_size_patch,image_size_patch)
            cls_resized = F.interpolate(cls_resized,(image_size,image_size), mode='bilinear', align_corners=False)
            cls_resized = cls_resized.view(image_size, image_size, 1)
            cls_resized = cls_resized.cpu().detach().numpy()[:, :, 0]
            cls_resized = median_filter(cls_resized, footprint=morphology.disk(patch_size // 2))
            cls_resized = gaussian_filter(cls_resized, sigma=patch_size // 4)
            true_label=mapping[label]
            pred_label=mapping[int(pred)]
            save_image(img,cls_resized,true_label,pred_label,i,arch)
            i+=1

def main():
    device,devname = identify_device(seed)
    arch=parse_arguments()
    
    print(f"================================================================")
    print(f"Using {device} - {devname}")
    print("Computing the attention maps for the 1^ fold")
    print(f"ViT - {arch}")
    print(f"================================================================\n")
    
    compute_att_maps(device,arch)
    exit(0)
    

if __name__=="__main__":
    main()
