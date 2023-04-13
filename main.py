import config as c
from train import train
from utils import load_datasets, make_dataloaders, get_loaders
import os
os.environ['TORCH_HOME'] = 'models\\EfficientNet'
# class_name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
#               'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for i in class_name:
    c.class_name = i
    # sensory dataset
    # train_set, test_set = load_datasets(c.dataset_path, c.class_name)
    # train_loader, test_loader = make_dataloaders(train_set, test_set)
    # semantic dataset
    train_loader,test_loader = get_loaders(c.dataset,c.class_name,c.batch_size)
    model = train(train_loader, test_loader)
    # model = eval(train_loader,test_loader)