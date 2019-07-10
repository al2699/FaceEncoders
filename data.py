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
      self.root_dir = root_dir
      tempDF = pd.read_csv(csv_file)
      for h in df.head():
         if(h != "image_path"):
            tempDF = tempDF.drop([h],axis=1)
      self.images = tempDF
      tempDF = pd.read_csv(csv_file)
      self.labels = tempDF.drop(["image_path"], axis=1)
      self.transform = transform

   #Returns the ith row of our dataframe
   def __getitem__(self, idx):
      #assuming no transformation
      image_path = self.images["image_path"][idx]
      image = cv2.imread(image_path)
      if(self.transform == True):
         image = self.transform(image)
      #MAYBE CHANGE TO FLOAT16 LATER
      label = np.asarray(self.labels.loc[idx].tolist(), dtype=np.float16)
      #return a tuple of image 
      return (image, label)

   def __len__(self):
      #must return the length of this dataset
      return len(self.images["image_path"])

"""Cohn-Kanade dataset class"""
class CKDataset(Dataset):
   def __init__(self, csv_file, root_dir, transform=None):
      """
      Args:
         csv_file (string): path to the CSV file
         root_dir (string): Directory to all of the images
         transform (callable, optional): Optional transform to be applied
                                         at retrieval time
      """
      self.root_dir = root_dir
      tempDF = pd.read_csv(csv_file)
      for h in df.head():
         if(h != "image_path"):
            tempDF = tempDF.drop([h],axis=1)
      self.images = tempDF
      tempDF = pd.read_csv(csv_file)
      self.labels = tempDF.drop(["image_path"], axis=1)
      self.transform = transform
,
   #Returns the ith row of our dataframe
   def __getitem__(self, idx):
      #assuming no transformation
      image_path = self.images["image_path"][idx]
      image = cv2.imread(image_path)
      if(self.transform == True):
         image = self.transform(image)
      #MAYBE CHANGE TO FLOAT16 LATER
      label = np.asarray(self.labels.loc[idx].tolist(), dtype=np.float16)
      #return a tuple of image 
      return (image, label)

   def __len__(self):
      #must return the length of this dataset
      return len(self.images["image_path"])

