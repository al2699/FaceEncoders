import os
import pandas as pd

frames = []

for root, dirs, files in os.walk("/data/deep/Alan/FacialEncodingDataset-OpenFace/BP4D"):
   for f in files:
      path = root + "/" + f
      if(f[-4:] == ".csv"):
         print("Processing: " + path)
         try:
            tempDF = pd.read_csv(path)
            tempDF["image_path"] = path.replace(".csv",".jpg")
            frames.append(tempDF)
         except:
            print("Opening " + path + " went wrong")
            continue
finalDF = pd.concat(frames)
finalDF.to_csv("BP4D.csv", index=False)
