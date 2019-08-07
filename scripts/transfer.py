import pandas as pd

#TODO: fill these in
train_csv = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/train.csv"
test_csv = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/test.csv"
valid_csv = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/valid.csv"

train_save = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/train_new.csv"
test_save = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/test_new.csv"
valid_save = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/valid_new.csv"

au6_dir = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/AU06/"
au10_dir = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/AU10/"
au12_dir = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/AU12/"
au14_dir = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/AU014/"
au17_dir = "/home/ICT2000/ahernandez/Downloads/AU-Intensity/AU17/"

def main():

   test_df = pd.read_csv(test_csv)
   #Here we add to the test df
   i = 0
   while i < len(test_df["image_path"]):
      print(str(i))
      split_array = []
      try:
         split_array = test_df["image_path"][i].split("/")
      except:
         print("In break")
         break
      first = split_array[4]
      second = split_array[5]
      #May need to turn into int
      imgNum = int(split_array[6][:-4])   
      au_dir = au6_dir + first + "_" + second + "_" + "AU06.csv"
      au6DF = pd.read_csv(au_dir)
      au10DF = pd.read_csv(au_dir.replace("AU06", "AU10"))
      au12DF = pd.read_csv(au_dir.replace("AU06", "AU12"))
      au14DF = pd.read_csv(au_dir.replace("AU06", "AU14"))
      au17DF = pd.read_csv(au_dir.replace("AU06", "AU17"))

      au6Dict = dict(zip(au6DF.iloc[:, 0], au6DF.iloc[:, 1]))
      au10Dict = dict(zip(au10DF.iloc[:, 0], au10DF.iloc[:, 1]))
      au12Dict = dict(zip(au12DF.iloc[:, 0], au12DF.iloc[:, 1]))
      au14Dict = dict(zip(au14DF.iloc[:, 0], au14DF.iloc[:, 1]))
      au17Dict = dict(zip(au17DF.iloc[:, 0], au17DF.iloc[:, 1]))

      try:
         test_df["AU06"][i] = au6Dict[imgNum]
         test_df["AU10"][i] = au10Dict[imgNum]
         test_df["AU12"][i] = au12Dict[imgNum]
         test_df["AU14"][i] = au14Dict[imgNum]
         test_df["AU17"][i] = au17Dict[imgNum]
         i += 1
      except:
         test_df = test_df.drop(test_df.index[i])
         test_df = test_df.reset_index()
         test_df = test_df.drop(["index"], axis=1)
         print(test_df)

   test_df.to_csv(test_save, index=False)


   print("Starting train move")
   train_df = pd.read_csv(train_csv)
   i = 0
   #Here we add to the test df
   while i < len(train_df["image_path"]):
      split_arry =[]
      print(str(i))
      try:
         split_array = train_df["image_path"][i].split("/")
      except:
         break
      first = split_array[4]
      second = split_array[5]
      #May need to turn into int
      imgNum = int(split_array[6][:-4])
   
      au_dir = au6_dir + first + "_" + second + "_" + "AU06.csv"
      au6DF = pd.read_csv(au_dir)
      au10DF = pd.read_csv(au_dir.replace("AU06", "AU10"))
      au12DF = pd.read_csv(au_dir.replace("AU06", "AU12"))
      au14DF = pd.read_csv(au_dir.replace("AU06", "AU14"))
      au17DF = pd.read_csv(au_dir.replace("AU06", "AU17"))

      au6Dict = dict(zip(au6DF.iloc[:, 0], au6DF.iloc[:, 1]))
      au10Dict = dict(zip(au10DF.iloc[:, 0], au10DF.iloc[:, 1]))
      au12Dict = dict(zip(au12DF.iloc[:, 0], au12DF.iloc[:, 1]))
      au14Dict = dict(zip(au14DF.iloc[:, 0], au14DF.iloc[:, 1]))
      au17Dict = dict(zip(au17DF.iloc[:, 0], au17DF.iloc[:, 1]))

      try:
         train_df["AU06"][i] = au6Dict[imgNum]
         train_df["AU10"][i] = au10Dict[imgNum]
         train_df["AU12"][i] = au12Dict[imgNum]
         train_df["AU14"][i] = au14Dict[imgNum]
         train_df["AU17"][i] = au17Dict[imgNum]
         i += 1
      except:
         train_df = train_df.drop(train_df.index[i])
         train_df = train_df.reset_index()
         train_df = train_df.drop(["index"], axis=1)

   train_df.to_csv(train_save, index=False)

   i = 0
   valid_df = pd.read_csv(valid_csv)
   #Here we add to the test df
   while i < len(valid_df["image_path"]):
      split_array = []
      try:      
         split_array = valid_df["image_path"][i].split("/")
      except:
         break
      first = split_array[4]
      second = split_array[5]
      #May need to turn into int
      imgNum = int(split_array[6][:-4])
   
      au_dir = au6_dir + first + "_" + second + "_" + "AU06.csv"
      au6DF = pd.read_csv(au_dir)
      au10DF = pd.read_csv(au_dir.replace("AU06", "AU10"))
      au12DF = pd.read_csv(au_dir.replace("AU06", "AU12"))
      au14DF = pd.read_csv(au_dir.replace("AU06", "AU14"))
      au17DF = pd.read_csv(au_dir.replace("AU06", "AU17"))

      au6Dict = dict(zip(au6DF.iloc[:, 0], au6DF.iloc[:, 1]))
      au10Dict = dict(zip(au10DF.iloc[:, 0], au10DF.iloc[:, 1]))
      au12Dict = dict(zip(au12DF.iloc[:, 0], au12DF.iloc[:, 1]))
      au14Dict = dict(zip(au14DF.iloc[:, 0], au14DF.iloc[:, 1]))
      au17Dict = dict(zip(au17DF.iloc[:, 0], au17DF.iloc[:, 1]))

      try:
         valid_df["AU06"][i] = au6Dict[imgNum]
         valid_df["AU10"][i] = au10Dict[imgNum]
         valid_df["AU12"][i] = au12Dict[imgNum]
         valid_df["AU14"][i] = au14Dict[imgNum]
         valid_df["AU17"][i] = au17Dict[imgNum]
         i += 1
      except:
         valid_df = valid_df.drop(valid_df.index[i])
         valid_df = valid_df.reset_index()
         valid_df = valid_df.drop(["index"], axis=1)

   valid_df.to_csv(valid_save, index=False)

if __name__ == "__main__":
   main()
