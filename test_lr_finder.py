import timm
import torch
import pandas as pd
import os
import multiprocessing as mp

from torchvision import transforms as T
from cassava_leaf_disease import CassavaLeafDiseaseDataset
from lr_finder import LRFinder
from learner import Learner
from sklearn import model_selection as ms
from classifier import Classifier

if __name__ == '__main__':
    device = torch.device("cpu")

    net = Classifier('tf_efficientnet_b4_ns', 5, pretrained=True)

    transform = T.Compose([T.ToTensor(),
                           T.Resize((380, 380)),
                           T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    data_root = "/home/namnd/personal-workspace/cassava-leaf-disease-classification"
    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    # train_df, val_df = ms.train_test_split(df, test_size=0.2, random_state=42, stratify=df.label.values)
    #
    # train_dataset = CassavaLeafDiseaseDataset(data_root, df=train_df, transform=transform)
    # val_dataset = CassavaLeafDiseaseDataset(data_root, df=val_df, transform=transform)
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=mp.cpu_count())
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=mp.cpu_count())

    dataloader = CassavaLeafDiseaseDataset(data_root, df, transform=transform)
    learner = Learner(net, dataloader, device)
    lr_finder = LRFinder(learner)
    lr_finder.find()
    lr_finder.plot()
