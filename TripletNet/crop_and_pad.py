import torchvision.transforms as t
from PIL import Image
import pandas as pd

#TODO: FILL THIS IN
csv_path = ""

#Helper function
def get_pil_image(img_path):
   toTensor = t.ToTensor()
   toPImage = t.ToPILImage()
   return toPImage(toTensor(img_path))

def get_crop_image(index, triplet_num):
   p_img = get_pil_image(img_path)
   df = pd.read_csv(csv_path)
   img_path = df["image"]
   #REMEMBER TO ADD NUMBER TO COEF HEADERS IN CSV
   tlc_coef = df["Top Left Column" + str(triplet_num)][index]
   blr_coef = df["Bottom Right Column" + str(triplet_num)][index]
   tlr_coef = df["Top Right Row" + str(triplet_num)][index]
   brr_coef = df["Bottom Right Row" + str(triplet_num)][index]

   #Calculate bounds
   width = p_img.size[0]
   height = p_img.size[1]
   left = int(tlc_coef * width) #normalized by width
   upper = int(tlr_coef * height) #normalized by height
   right = int(brc_coef * width)
   lower = int(brr_coef * height)
   box = (left, upper, right, lower)
   
   #Finally, crop
   crop_img = p_img.crop(box)
   return crop_img

def pad_iamge(p_img):
   aspect_ratio = p_img.size[0] / p_img.size[1]
   one_over_ar = p_img.size[0] / p_img.size[1]
   if(aspect_ratio < 1):
      #Need to pad width
      
   elif(one_over_ar < 1):
      #Need to pad height
   else:
      #already square image
      return p_img

   if side == "height":
      #Then we pad the the height until height == width
            
   else:
      #Then we pad the width until height == width
df = pd.read_csv(csv_path)


