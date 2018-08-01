import torch
import torchvision
import torchvision.transforms as transforms
from datasets import camvid
import utils.imgs
import utils.training as train_utils
from datasets import joint_transforms
from pathlib import Path
from models import tiramisu

CAMVID_PATH = Path('/home/jingwenlai/data', 'CamVid/CamVid')
batch_size = 2

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
test_dset = camvid.CamVid(CAMVID_PATH, 'test', joint_transform=None,
transform=transforms.Compose([
    transforms.ToTensor(),
    normalize
]))
test_loader = torch.utils.data.DataLoader(test_dset, batch_size = batch_size, shuffle=False)

print("Test: %d"%len(test_loader.dataset.imgs))

model = tiramisu.FCDenseNet67(n_classes=12).cuda()
model_weights = ".weights/latest.th"
startEpoch = train_utils.load_weights(model, model_weights)
print("load_weights, return epoch: ", startEpoch)


train_utils.view_sample_predictions(model, test_loader, n=10)
