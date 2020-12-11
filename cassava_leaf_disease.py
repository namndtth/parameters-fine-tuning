import torch
import os
from PIL import Image


class CassavaLeafDiseaseDataset(torch.utils.data.Dataset):
    """Cassava leaf disease dataset."""

    def __init__(self, root_dir, subset="train_images", df=None, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.subset = "train_images"
        self.df_data = df.values

        if self.subset not in ["train_images", "test_images"]:
            raise TypeError("Subset must be in train_images or test_images but your is " + self.subset)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name, label = self.df_data[idx]
        img_path = os.path.join(self.root_dir, self.subset, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
