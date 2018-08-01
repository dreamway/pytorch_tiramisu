# coding: utf-8

# ## Dependencies

import os
import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils

import matplotlib.pyplot as plt



# ## CamVid
# 
# Clone this repository which holds the CamVid dataset
# ```
# git clone https://github.com/alexgkendall/SegNet-Tutorial
# ```

# In[1]:
CAMVID_PATH = Path('/home/jingwenlai/data', 'CamVid/CamVid')
print("CAMVID_PATH:",CAMVID_PATH)
RESULTS_PATH = Path('.results/')
WEIGHTS_PATH = Path('.weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 1

best_loss = float('inf')

# In[3]:

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose([
    #joint_transforms.JointRandomCrop(224), # commented for fine-tuning
    joint_transforms.JointRandomHorizontalFlip()
    ])
train_dset = camvid.CamVid(CAMVID_PATH, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.ToTensor(),
          normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size, shuffle=True)

val_dset = camvid.CamVid(
    CAMVID_PATH, 'val', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=batch_size, shuffle=False)

test_dset = camvid.CamVid(
    CAMVID_PATH, 'test', joint_transform=None,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False)


# In[4]:


print("Train: %d" %len(train_loader.dataset.imgs))
print("Val: %d" %len(val_loader.dataset.imgs))
print("Test: %d" %len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

#utils.imgs.view_image(inputs[0])
#utils.imgs.view_annotated(targets[0])


# ## Train

# In[5]:


LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 100
torch.cuda.manual_seed(0)


# In[6]:


#model = tiramisu.FCDenseNet67(n_classes=12).cuda()
model = tiramisu.FCDenseNet103(n_classes=12).cuda()
model.apply(train_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.NLLLoss2d(weight=camvid.class_weight.cuda()).cuda()


# In[7]:
def train(epoch):
    since = time.time()
    ### Train ###
    trn_loss, trn_err = train_utils.train(model, train_loader, optimizer, criterion, epoch)
    #print('Epoch {:d}, Train - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, trn_loss, 1-trn_err))    
    print("Epoch: %d, Train - Loss: %.4f, Acc: %.4f"%(epoch, trn_loss, 1-trn_err))
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return trn_loss, 1-trn_err


def val(epoch):
    since = time.time()
    ### Test ###
    val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)    
    print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, 1-val_err))
    time_elapsed = time.time() - since  
    print('Total Time {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))

    global best_loss
    if val_loss < best_loss:
        ### Checkpoint ###    
        train_utils.save_weights(model, epoch, val_loss, val_err)
        best_loss = val_loss

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, 
                                     epoch, DECAY_EVERY_N_EPOCHS)

    return val_loss, 1-val_err


# ## Test
def test():
    train_utils.test(model, test_loader, criterion, epoch=1)  
    train_utils.view_sample_predictions(model, test_loader, n=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pytorch Tiramisu Training")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--resume', '-r', action="store_true", help="resume from checkpoint")
    args = parser.parse_args()

    startEpoch = 0
    if args.resume:
        startEpoch = train_utils.load_weights(model,'.weights/latest.th')
        print("load weights for model , start from startEpoch")
        train(startEpoch)
    
    epoch_logger = open('train_val_log.csv','w')

    for epoch in range(startEpoch, N_EPOCHS+1):  
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = val(epoch)

        epoch_logger.write("%d, %.3f, %.3f, %.3f, %3f"%(epoch, train_loss, train_acc, val_loss, val_acc))

    epoch_logger.close()
