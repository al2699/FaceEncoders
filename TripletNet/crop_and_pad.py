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

def pad_image(p_img):
   desired_size = 224
   old_size = p_img.size  # old_size[0] is in (width, height) format

   ratio = float(desired_size)/max(old_size)
   new_size = tuple([int(x*ratio) for x in old_size])
   # use thumbnail() or resize() method to resize the input image

   # thumbnail is a in-place operation

   # im.thumbnail(new_size, Image.ANTIALIAS)

   im = p_img.resize(new_size, Image.ANTIALIAS)
   # create a new image and paste the resized on it

   new_im = Image.new("RGB", (desired_size, desired_size))
   new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))

   new_im.show()

if __name__ == "__main__":
   p_img = get_crop_image(0, 1)
   pad_image(p_img)
