import data
import random
import pandas as pd

w300_split_path = "/data/deep/Alan/FacialEncodingDataset-OpenFace/300W-Processed/split.csv"
ck_split_path = "/data/deep/Alan/FacialEncodingDataset-OpenFace/CK+-Processed/split.csv"
bp4d_split_path = "/data/deep/Alan/FacialEncodingDataset-OpenFace/BP4D/split.csv"

def train_test_validate_split(dataset):
   arr = range(0, len(dataset))
   arr = list(arr)
   #10% valid, 10% testing, 80% training
   test_amount = int(.10 * len(dataset))
   train_indices = []
   test_indices = []
   validate_indices = []

   #consistent seed for all split
   random.seed(42)
   for i in range(test_amount):
      pick = random.randint(0, len(arr) - 1)
      test_indices.append(arr[pick])
      del arr[pick]

   for i in range(test_amount):
      pick = random.randint(0, len(arr) - 1)
      validate_indices.append(arr[pick])
      del arr[pick]

   train_indices = arr

   return train_indices, test_indices, validate_indices

if __name__ == "__main__":
   w300 = data.W300Dataset()
   ck = data.CKDataset()
   bp4d = data.BP4DDataset()

   train, test, valid = train_test_validate_split(w300)
   d = {"train" : train, "test" : test, "validation" : valid}
   tempDF = pd.DataFrame.from_dict(d, orient="index").transpose()
   tempDF.to_csv(w300_split_path, index=False)

   train, test, valid = train_test_validate_split(ck)
   d = {"train" : train, "test" : test, "validation" : valid}
   tempDF = pd.DataFrame.from_dict(d, orient="index").transpose()
   tempDF.to_csv(ck_split_path, index=False)

   train, test, valid = train_test_validate_split(bp4d)
   d = {"train" : train, "test" : test, "validation" : valid}
   tempDF = pd.DataFrame.from_dict(d, orient="index").transpose()
   tempDF.to_csv(bp4d_split_path, index=False)
   
