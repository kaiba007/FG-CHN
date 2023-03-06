import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datacsv.fg_dataset import FG_dataset
from torchvision import transforms

class DInterface(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.train_data = FG_dataset((self.config.train_csv,), self.config.data_root, config=self.config, transform=train_transform)
        self.val_hashing_data = FG_dataset((self.config.train_csv, self.config.test_csv), self.config.data_root, data_type='hashing', transform=test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_hashing_data, batch_size=32, shuffle=False, pin_memory=False, num_workers=self.config.num_workers)

