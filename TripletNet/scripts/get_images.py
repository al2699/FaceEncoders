import pandas as pd
import urllib.request
#import dlib #for face detector/ensuring valid data
import time
import os

savePath = "/home/ICT2000/ahernandez/Downloads/GoogleDataset-Test/"
csvPath = "/home/ICT2000/ahernandez/Downloads/fec_test.csv"
skippedFilePath = "/home/ICT2000/ahernandez/Downloads/GoogleDataset/skipped_test.txt"
df = pd.read_csv(csvPath)
#faceDetector = dlib.get_frontal_face_detector()
skipped = []
"""
def is_face(imagePath):
   img = dlib.load_rgb_image(imagePath)
   dets = faceDetector(img)
   if(len(dets) != 0):
      return True
   return False
"""
#assuming we have already given headers to the google dataset csv
start = time.time()
imgNum = 0
size = len(df["image1"])
imgMap = {} 
for i in range(0, size - 1):
#for i in range(0, 10):
   img1Link = df["image1"][i]
   img2Link = df["image2"][i]
   img3Link = df["image3"][i]
   #if imgLink is not in map:
      #add it to the map
      #we download the image
      #if the image doesn't contain a face
         #than we raise exception
      #else
         #write new local path to csv file
      #catch exception:
         #append the skipped link to skipped.txt
   #else:
      #get img lcoal link from the map
      #add img local link to the csv 
   if(not img1Link in imgMap):
      imgDir = savePath + str(imgNum) + ".jpg"
      imgNum += 1
      imgMap.update({img1Link : imgDir})
      try:
         urllib.request.urlretrieve(img1Link, imgDir)
         #if(not is_face(imgDir)):
            #raise Exception("not face")
         df["image1"][i] = imgDir
      except:
         skipped.append(str(i + 2) + ";" + img1Link + ";" + imgDir + "\n")
   else:
      imgDir = imgMap[img1Link]
      df["image1"][i] = imgDir

   if(not img2Link in imgMap):
      imgDir = savePath + str(imgNum) + ".jpg"
      imgNum += 1
      imgMap.update({img2Link : imgDir})
      try:
         urllib.request.urlretrieve(img2Link, imgDir)
         #if(not is_face(imgDir)):
            #raise Exception("not face")
         df["image2"][i] = imgDir
      except:
         skipped.append(str(i + 2) + ";" + img2Link + ";" + imgDir + "\n")
   else:
      imgDir = imgMap[img2Link]
      df["image2"][i] = imgDir

   if(not img3Link in imgMap):
      imgDir = savePath + str(imgNum) + ".jpg"
      imgNum += 1
      imgMap.update({img3Link : imgDir})
      try:
         urllib.request.urlretrieve(img3Link, imgDir)
         #if(not is_face(imgDir)):
            #raise Exception("not face")
         df["image3"][i] = imgDir
      except:
         skipped.append(str(i + 2) + ";" + img3Link + ";" + imgDir + "\n")
   else:
      imgDir = imgMap[img3Link]
      df["image3"][i] = imgDir
   if(i % 5000 == 0):
      print("At " + str(i) + "th row")
   #face detection portion
   #after face detection/verification and retrieval
   #IF EVERYTHING WAS SAVED WELL:
f = open(savePath + "skipped_test.txt", "w")
for entry in skipped:
   f.write(entry)
f.close()
df.to_csv(csvPath)

end = time.time()
print("Done. Time: " + str(end - start))
