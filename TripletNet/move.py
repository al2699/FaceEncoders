#!/home/ICT2000/ahernandez/anaconda3/envs/myenv/bin/python3

import os
import pandas as pd

fec_path = "/data1/Alan/GoogleDataset/fec_train_new.csv"
train_path = "/data1/Alan/GoogleDataset/train.csv"
valid_path = "/data1/Alan/GoogleDataset/valid.csv"

split_df = pd.read_csv("/data1/Alan/GoogleDataset/split.csv")
fec_df = pd.read_csv(fec_path)

def get_length(d_df, indices):
   size = 0
   for i in range(len(indices)):
      try:
         d_df["image_path"][indices[i]]
         size += 1
      except:
         #reached the end
         return size

train_size = len(split_df["train"])
valid_size = int(train_size * 0.1)

print("Starting retrieval of train")
train_frames = []
#Move and create train
for i in range(train_size):
   s = fec_df.loc[split_df["train"][i]]
   temp_df = pd.DataFrame([s.rename(None)])
   train_frames.append(temp_df)
   if i % 1000 == 0:
      print(str(i / train_size) + "Done")

train_df = pd.concat(train_frames)

"""
print("Starting retrieval of test")
test_frames = []
for i in range(test_size):
   s = fec_df.loc[split_df["test"][i]]
   temp_df = pd.DataFrame([s.rename(None)])
   test_frames.append(temp_df)
   if i % 1000 == 0:
      print(str(i / test_size) + "Done")

test_df = pd.concat(test_frames)
"""
print("Starting retrieval of validation")
validation_frames = []
for i in range(valid_size):
   s = fec_df.loc[split_df["validation"][i]]
   temp_df = pd.DataFrame([s.rename(None)])
   validation_frames.append(temp_df)
   if i % 1000 == 0:
      print(str(i / valid_size) + "Done")

validation_df = pd.concat(validation_frames)

train_df.to_csv(train_path, index=False)
#test_df.to_csv(test_path, index=False)
validation_df.to_csv(valid_path, index=False)
