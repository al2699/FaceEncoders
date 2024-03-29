#!/home/ICT2000/ahernandez/anaconda3/envs/myenv/bin/python3
import data
import random
import pandas as pd

save_path = "/data1/Alan/GoogleDataset-Default/split.csv"
fec_csv_path = "/data1/Alan/GoogleDataset-Default/fec_train.csv"

def train_validate_split(dataset):
   arr = range(0, len(dataset))
   arr = list(arr)
   #10% valid, 10% testing, 80% training
   test_amount = int(.10 * len(dataset))
   train_indices = []
   validate_indices = []

   #consistent seed for all split
   random.seed(42)
   for i in range(test_amount):
      print("At: " + str(i))
      pick = random.randint(0, len(arr) - 1)
      validate_indices.append(arr[pick])
      del arr[pick]

   train_indices = arr

   return train_indices, validate_indices

if __name__ == "__main__":
   fec = data.FECDataset(csv_file=fec_csv_path) #Set to train set by default   

   train, valid = train_validate_split(fec)
   d = {"train" : train, "validation" : valid}
   tempDF = pd.DataFrame.from_dict(d, orient="index").transpose()
   tempDF.to_csv(save_path, index=False)
