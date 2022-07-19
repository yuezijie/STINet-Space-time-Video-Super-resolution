from torchvision.transforms import Compose, ToTensor, Normalize
from dataset_STINet import *

def transform():
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

##LOADER TRAINING
def get_training_set(data_dir, upscale_factor, data_augmentation, file_list):
    return DatasetFromFolder(data_dir, upscale_factor, data_augmentation, file_list,transform=transform())



