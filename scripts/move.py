import os
import pandas as pd

bp4d_path = "/data1/Alan/BP4D/BP4D_cropped.csv"
train_path = "/data1/Alan/BP4D/train.csv"
test_path = "/data1/Alan/BP4D/test.csv"
valid_path = "/data1/Alan/BP4D/valid.csv"

split_df = pd.read_csv("split.csv")
bp4d_df = pd.read_csv(bp4d_path)

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
test_size = get_length(bp4d_df, split_df["test"])
valid_size = get_length(bp4d_df, split_df["validation"])

print("Starting retrieval of train")
train_frames = []
#Move and create train
for i in range(train_size):
   s = bp4d_df.loc[split_df["train"][i]]
   temp_df = pd.DataFrame([s.rename(None)])
   train_frames.append(temp_df)
   if i % 1000 == 0:
      print(str(i / train_size) + "Done")

train_df = pd.concat(train_frames)

print("Starting retrieval of test")
test_frames = []
for i in range(test_size):
   s = bp4d_df.loc[split_df["test"][i]]
   temp_df = pd.DataFrame([s.rename(None)])
   test_frames.append(temp_df)
   if i % 1000 == 0:
      print(str(i / train_size) + "Done")

test_df = pd.concat(test_frames)

print("Starting retrieval of validation")
validation_frames = []
for i in range(valid_size):
   s = bp4d_df.loc[split_df["validation"][i]]
   temp_df = pd.DataFrame([s.rename(None)])
   validation_frames.append(temp_df)
   if i % 1000 == 0:
      print(str(i / train_size) + "Done")

validation_df = pd.concat(validation_frames)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
validation_df.to_csv(valid_path, index=False)
