import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Dataloader
from torchvision import transforms, utils

#Dataset paths
W300_CSV = "/data/deep/Alan/FacialEncodingDataset-OpenFace/300W-Processed/300W.csv"
CK_CSV = "/data/deep/Alan/FacialEncodingDataset-OpenFace/Ck+-Processed/CK+.csv"


"""300 faces in the wild dataset class"""
class W300Dataset(Dataset):
   def __init__(self, csv_file, root_dir, transform=None):
      """
      Args:
         csv_file (string): path to the CSV file
         root_dir (string): Directory to all of the images
         transform (callable, optional): Optional transform to be applied
                                         at retrieval time
      """
      self.root_dir = root_dor
      self.df = pd.read_csv(csv_file)
      self.transform = transform

   #Returns the ith row of our dataframe
   def __getitem__(self, idx):
      #assuming no transformation
      return self.df.loc[idx]

   def __len__(self):
      #must return the length of this dataset
      return len(self.df["frame"])

"""Cohn-Kanade plus dataset class"""
class CKDataset(Dataset):
   def __init__(self, csv_file, root_dir, transform=None):
      """
      Args:
         csv_file (string): path to the CSV file
         root_dir (string): Directory to all of the images
         transform (callable, optional): Optional transform to be applied
                                         at retrieval time
      """
      self.root_dir = root_dor
      self.df = pd.read_csv(csv_file)
      self.transform = transform

   #Returns the ith row of our dataframe
   def __getitem__(self, idx):
      #assuming no transformation
      return self.df.loc[idx]

   def __len__(self):
      #must return the length of this dataset
      return len(self.df["frame"])

"""Cohn-Kanade plus dataset class"""
class CKDataset(Dataset):
   def __init__(self, csv_file, root_dir, transform=None):
      """
      Args:
         csv_file (string): path to the CSV file
         root_dir (string): Directory to all of the images
         transform (callable, optional): Optional transform to be applied
                                         at retrieval time
      """
      self.root_dir = root_dor
      self.df = pd.read_csv(csv_file)
      self.transform = transform

   #Returns the ith row of our dataframe
   def __getitem__(self, idx):
      #assuming no transformation
      return self.df.loc[idx]

   def __len__(self):
      #must return the length of this dataset
      return len(self.df["frame"])
