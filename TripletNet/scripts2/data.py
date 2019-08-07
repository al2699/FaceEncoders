import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import cv2
from PIL import Image
import statistics

fec_train = "/data1/Alan/BP4D/train.csv"
fec_test = "/data1/Alan/BP4D/test.csv"
fec_valid = "/data1/Alan/BP4D/valid.csv"

#Margin map
margin_map = {"ONE_CLASS_TRIPLET" : 0.1, "TWO_CLASS_TRIPLET" : 0.2, "THREE_CLASS_TRIPLET" : 0.2}

def get_mode(lis):
   mode = None
   try:
      mode = statistics.mode(lis)
   except:
      onesL = 0
      twosL = 0
      threesL = 0
      for num in lis:
         if num == 1:
            onesL += 1
         elif num == 2:
            twosL += 1
         else:
            threesL += 1
      tempDict = {onesL : 1, twosL : 2, threesL : 3}
      mode = tempDict[max(onesL, twosL, threesL)]
   return mode

"""Facial Expression Comparison dataset class"""
class FECDataset(Dataset):
   #Uses train dataset csv by default. Can createa test dataset by passing in path.
   def __init__(self, csv_file=None, transform=None):
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
         ann_key = "Annotation" + str(i)
         votes.append(int(row[ann_key]))
      #print("Choosing from: " + str(votes))
      furthest_img_ind = get_mode(votes)
     
      #Extract images and label them based on whether or not they are
      #similar to each other
      img3 = Image.open(row["image" + str(furthest_img_ind)])
      img_options.remove(furthest_img_ind)
      img1 = Image.open(row["image" + str(img_options[0])])
      img2 = Image.open(row["image" + str(img_options[1])])
      #print("Extracted: " + image_path)

      #Perform transforma/normalization to fit what resnet pretrained
      #model was trained on
      if self.transform is not None:
         img1 = self.transform(img1)
         img2 = self.transform(img2)
         img3 = self.transform(img3)

      return (img1, img2, img3, margin)

   def __len__(self):
      #must return the length of this dataset
      return len(self.df["image1"])
