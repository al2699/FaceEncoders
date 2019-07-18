import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import cv2

#Dataset paths
W300_CSV = "/data/deep/Alan/FacialEncodingDataset-OpenFace/300W-Processed/300W_cropped.csv"
CK_CSV = "/data/deep/Alan/FacialEncodingDataset-OpenFace/CK+-Processed/CK+_cropped.csv"
#BP4D CSV still not made
BP4D_CSV = "/data/deep/Alan/FacialEncodingDataset-OpenFace/BP4D/BP4D_cropped.csv"

#Split lists
W300_split_list = "/data/deep/Alan/FacialEncodingDataset-OpenFace/300W-Processed/split.csv"
CK_split_list = "/data/deep/Alan/FacialEncodingDataset-OpenFace/CK+-Processed/split.csv"
BP4D_split_list = "/data/deep/Alan/FacialEncodingDataset-OpenFace/BP4D/split.csv"

#TODO: Could create one master class with the general fomat of W300, 
#BP4D, CK+ and later inherit from it
"""300 faces in the wild dataset class"""
class W300Dataset(Dataset):
   def __init__(self, csv_file=W300_CSV, transform=None):
      """
      Args:
         csv_file (string): path to the CSV file
         root_dir (string): Directory to all of the images
         transform (callable, optional): Optional transform to be applied
                                         at retrieval time
      """
      #self.root_dir = root_dir
      tempDF = pd.read_csv(csv_file)
      for h in tempDF.head():
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
      #print("Extraced: " + image_path)
      if(image.all() == None):
         print("Missing: " + image_path)
      if(self.transform == True):
         image = self.transform(image)
      image_tensor = torch.Tensor(image)
      image = Variable(image_tensor, requires_grad=False)
      #MAYBE CHANGE TO FLOAT16 LATER
      label = np.asarray(self.labels.loc[idx].tolist(), dtype=np.float16)
      label_tensor = torch.Tensor(label).view(-1,1)
      label = Variable(label_tensor, requires_grad=False)
      #return a tuple of image 
      return (image, label)

   def __len__(self):
      #must return the length of this dataset
      return len(self.images["image_path"])
   
   """
   Returns a list of the indices which include the rows of data for each
   respective portion of the split
   def train_test_validation_split(self):
      tempDF = pd.read_csv(W300_split_list)
      train_list = tempDF["train"].tolist()
      test_list = tempDF["test"].tolist()
      valid_list = tempDF["validation"].tolist()

      return train_list, test_list, valid_list
   """

"""Cohn-Kanade dataset class"""
class CKDataset(Dataset):
   def __init__(self, csv_file=CK_CSV, transform=None):
      """
      Args:
         csv_file (string): path to the CSV file
         root_dir (string): Directory to all of the images
         transform (callable, optional): Optional transform to be applied
                                         at retrieval time
      """
      #self.root_dir = root_dir
      print(csv_file)
      tempDF = pd.read_csv(csv_file)
      for h in tempDF.head():
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
      #print("Extracted: " + image_path)
      if(image.all() == None):
         print("Missing: " + image_path)
      if(self.transform == True):
         image = self.transform(image)
      image_tensor = torch.Tensor(image)
      image = Variable(image_tensor, requires_grad=False)
      #MAYBE CHANGE TO FLOAT16 LATER
      label = np.asarray(self.labels.loc[idx].tolist(), dtype=np.float16)
      label_tensor = torch.Tensor(label).view(-1,1)
      label = Variable(label_tensor, requires_grad=False)
      #return a tuple of image 
      return (image, label)

   def __len__(self):
      #must return the length of this dataset
      return len(self.images["image_path"])

   """
   def train_test_validation_split(self):
      tempDF = pd.read_csv(W300_split_list)
      train_list = tempDF["train"].tolist()
      test_list = tempDF["test"].tolist()
      valid_list = tempDF["validation"].tolist()

      return train_list, test_list, valid_list
   """
