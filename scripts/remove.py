import pandas as pd
import cv2

df = pd.read_csv("BP4D_cropped.csv")
newDF = df
badCount = 0

for i in range(len(df["image_path"])):
   img = cv2.imread(df["image_path"][i], 0)
   if i % 1000 == 0: print("Looking at: " + df["image_path"][i])
   if cv2.countNonZero(img) == 0:
      print("Found bad at: " + str(i))
      badCount += 1
i = 0
j = 0
noBadImg = True
print(df)
while(j != badCount):
   while(noBadImg):
      print("Before opening img: " + str(i))
      img = cv2.imread(df["image_path"][i], 0)
      if cv2.countNonZero(img) == 0:
         print("Removing: " + str(i))
         df = df.drop(df.index[i])
         df = df.reset_index()
         df = df.drop(["index"], axis=1)
         i = 0
         j += 1
         noBadImg = False
      else:
         i += 1
   noBadImg = True
df.to_csv("BP4D_final.csv", index=False)
