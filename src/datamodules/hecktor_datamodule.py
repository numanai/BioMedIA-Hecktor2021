from typing import Optional, Tuple
from math import  pi

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

from src.datamodules.transforms import *

from src.datamodules.datasets.hecktor_dataset import HecktorDataset


class HECKTORDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = self.hparams["data_dir"]
        self.train_val_test_split = self.hparams["train_val_test_split"]
        self.batch_size = self.hparams["batch_size"]
        self.num_workers = self.hparams["num_workers"]
        self.pin_memory = self.hparams["pin_memory"]

        self.test_transforms = transforms.Compose(
            [ Normalize(self.hparams["dataset_mean"], self.hparams["dataset_std"]),
              ToTensor()]
        )

        self.train_transforms = transforms.Compose([
            RandomInPlaneRotation(pi / 6),
            RandomFlip(0),
            # RandomFlip(1),
            RandomFlip(2),
            Normalize(self.hparams["dataset_mean"], self.hparams["dataset_std"]),
            RandomNoise(.05),
            ToTensor(),
        ])



        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        self.dataset = HecktorDataset(self.hparams["root_dir"],
                                      self.hparams["data_dir"],
                                      self.hparams["patch_size"],
                                      transform=self.train_transforms,
                                      cache_dir=self.hparams["cache_dir"],
                                      num_workers=self.hparams["num_workers"])

    def setup(self, stage: Optional[str] = None):

        self.dataset = HecktorDataset(self.hparams["root_dir"],
                                      self.hparams["data_dir"],
                                      self.hparams["patch_size"],
                                      transform=self.train_transforms,
                                      cache_dir=self.hparams["cache_dir"],
                                      num_workers=self.hparams["num_workers"])
        
        full_indices = range(len(self.dataset))
        train_indices, val_indices, test_indices = random_split(
            full_indices, self.train_val_test_split
        )
        train_dataset, val_dataset, test_dataset = Subset(self.dataset, train_indices), Subset(self.dataset, val_indices), Subset(self.dataset, test_indices)
        val_dataset.dataset.transform = self.test_transforms
        test_dataset.dataset.transform = self.test_transforms

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
