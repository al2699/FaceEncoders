import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import cv2
import statistics

#Dataset paths
fec_csv_train = "/data1/Alan/GoogleDataset/fec_train_new.csv"
fec_csv_test = "/data1/Alan/GoogleDataset/fec_test_new1.csv"
split_path = "/data1/Alan/GoogleDataset/split.csv"

#Margin map
margin_map = {"ONE_CLASS_TRIPLET" : 0.1, "TWO_CLASS_TRIPLET" : 0.2, "THREE_CLASS_TRIPLET" : 0.2}

#Transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

"""Facial Expression Comparison dataset class"""
class FECDataset(Dataset):
   #Uses train dataset csv by default. Can createa test dataset by passing in path.
   def __init__(self, csv_file=fec_csv_train, transform=normalize):
      """
      Args:
         csv_file (string): path to the CSV file
         root_dir (string): Directory to all of the images
         transform (callable, optional): Optional transform to be applied
                                         at retrieval time
      """
      #self.root_dir = root_dir
      self.df = pd.read_csv(csv_file)
      self.transform = transform

   #Returns the ith row of our dataframe
   def __getitem__(self, idx):
      row = self.df.loc[idx]
      #Calculate margin based on triplet type
      margin = margin_map[row["Triplet_type"]]
      #Base similarity on group rating's mode
      votes = []
      img_options = [1,2,3]
      for i in range(1, 7):
         ann_key = "Annotator" + str(i) + "_id"
         votes.append(int(row[ann_key]))
      furthest_img_ind = statistics.mode(votes)
     
      #Extract images and label them based on whether or not they are
      #similar to each other
      img3 = cv2.imread(row["image" + str(furthest_img_ind)])
      img_options.remove(furthest_img_ind)
      img1 = cv2.imread(row["image" + str(img_options[0])])
      img2 = cv2.imread(row["image" + str(img_options[1])])
      #print("Extracted: " + image_path)

      #Perform transforma/normalization to fit what resnet pretrained
      #model was trained on
      if(self.transform == True):
         img1 = self.transform(img1)
         img2 = self.transform(img2)
         img3 = self.transform(img3)

      #Wrap images in variables
      im1_tensor = torch.Tensor(img1)
      im2_tensor = torch.Tensor(img2)
      im3_tensor = torch.Tensor(img3)
      img1_var = Variable(im1_tensor, requires_grad=False)
      img2_var = Variable(im2_tensor, requires_grad=False)
      img3_var = Variable(im3_tensor, requires_grad=False)

      return (img1_var, img2_var, img3_var, margin)

   def __len__(self):
      #must return the length of this dataset
      return len(self.df["image1"])
   
   #Returns a list of the indices which include the rows of data for each
   #respective portion of the split
   def train_valid_split(self):
      tempDF = pd.read_csv(split_path)
      valid = tempDF["validation"].tolist()
      train = tempDF["train"].tolist()

      return train, valid
