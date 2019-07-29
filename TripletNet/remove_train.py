#!/home/ICT2000/ahernandez/anaconda3/envs/myenv/bin/python3
import pandas as pd

df = pd.read_csv("/home/ICT2000/ahernandez/Downloads/faceexp-comparison-data-train-public.csv")
newDF = df
badCount = 0
dead_links = []
i = 0
#while i < 10:
while i < len(df["image1"]):
   if not("/home/ICT2000/" in df["image1"][i] and \
         "/home/ICT2000/" in df["image2"][i] and \
         "/home/ICT2000/" in df["image3"][i]):
      badCount += 1
      if not "/home/ICT2000/" in df["image1"][i]:
         dead_links.append(str(i + 2) + ": " + df["image1"][i])
      if not "/home/ICT2000/" in df["image2"][i]:
         dead_links.append(str(i + 2) + ": " + df["image2"][i])
      if not "/home/ICT2000/" in df["image3"][i]:
         dead_links.append(str(i + 2) + ": " + df["image3"][i])
      print("Removing row: " + str(i + 2))
      df = df.drop(df.index[i])
      df = df.reset_index()
      df = df.drop(["index"], axis=1)
      i -= 1
   i += 1

f = open("dl.txt", "w")
for link in dead_links:
   f.write(link + "\n")
f.close()
dead_links = []
print("Got bad count of: " + str(badCount))

df.to_csv("/home/ICT2000/ahernandez/Downloads/fec_train_non_dead.csv", index=False)
