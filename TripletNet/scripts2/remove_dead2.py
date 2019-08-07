#!/home/ICT2000/ahernandez/anaconda3/envs/myenv/bin/python3
import pandas as pd
from PIL import Image

csv_path = "/home/ICT2000/ahernandez/Downloads/fec_test.csv"
df = pd.read_csv(csv_path)

print("Beginning non existing img removal")
bad = 0
i = 0
while i < len(df["image1"]):
   try:
      Image.open(df["image1"][i])
      Image.open(df["image2"][i])
      Image.open(df["image3"][i])
      i += 1
   
   except:
      print("Removing: " + str(df["image1"][i]) + " or " + str(df["image2"][i]) + \
         " or " + str(df["image3"][i]))
      df = df.drop(df.index[i])
      df = df.reset_index()
      df = df.drop(["index"], axis=1)
      bad += 1
      i -= 1
print("Got bad of: " + str(bad))
df.to_csv("/home/ICT2000/ahernandez/Downloads/fec_test_good.csv", index=False)
