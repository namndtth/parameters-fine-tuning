import timm
import torch
import pandas as pd
import os
import multiprocessing as mp

from torchvision import transforms as T
from cassava_leaf_disease import CassavaLeafDiseaseDataset
from sklearn import model_selection as ms
from tqdm import tqdm

def train_one_epoch(net, criterion, optimizer, train_loader):
    epoch_loss = 0.0
    epoch_acc = 0.0

    for i, data in enumerate(tqdm(train_loader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        acc = (outputs.argmax(dim=1) == labels).float().mean()

        epoch_loss += loss.item()
        epoch_acc += acc

        optimizer.step()

        if (i % 100 == 99):
            print('training loss ', loss.item())
    print('epoch loss ', epoch_loss / len(train_loader), ' epoch acc ', epoch_acc)
    torch.save(vit16.state_dict(), './models/best.pth')


def val_one_epoch(net, val_loader):
    epoch_loss = 0.0
    epoch_acc = 0.0

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc

            if (i % 100 == 99):
                print('val loss ', loss.item())
        print('epoch loss ', epoch_loss / len(train_loader), ' epoch acc ', epoch_acc)


if __name__ == '__main__':
    device = torch.device("cpu")
    vit_model_names = timm.list_models('vit*')
    vit16 = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=5).to(device)

    transform = T.Compose([T.ToTensor(),
                           T.Resize((384, 384)),
                           T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    data_root = '/newDriver/nam/cassava-leaf-disease'
    working_root = './'

    df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    train_df, val_df = ms.train_test_split(df, test_size=0.2, random_state=42, stratify=df.label.values)

    train_dataset = CassavaLeafDiseaseDataset(data_root, df=train_df, transform=transform)
    val_dataset = CassavaLeafDiseaseDataset(data_root, df=val_df, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=mp.cpu_count())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=mp.cpu_count())

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit16.parameters(), lr=1e-3)

    for epoch in range(10):
        train_one_epoch(vit16, criterion, optimizer, train_loader)
        val_loader(vit16, val_loader)