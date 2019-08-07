#!/home/ICT2000/ahernandez/anaconda3/envs/myenv/bin/python3
import torchvision.transforms as t
from PIL import Image
import pandas as pd

#TODO: FILL THIS IN
csv_path = "/home/ICT2000/ahernandez/Downloads/fec_train_non_dead1.csv"
save_path = "/home/ICT2000/ahernandez/Downloads/fec_train_new_crop.csv"

#Helper function
def get_pil_image(img_path):
   img = Image.open(img_path)
   toTensor = t.ToTensor()
   toPImage = t.ToPILImage()
   i = toPImage(toTensor(img))
   return i

def get_crop_image(index, triplet_num):
   df = pd.read_csv(csv_path)
   img_path = df["image" + str(triplet_num)][index]
   p_img = get_pil_image(img_path)
   #REMEMBER TO ADD NUMBER TO COEF HEADERS IN CSV
   tlc_coef = df["Top Left Column" + str(triplet_num)][index]
   brc_coef = df["Bottom Right Column" + str(triplet_num)][index]
   tlr_coef = df["Top Left Row" + str(triplet_num)][index]
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

def pad_save_image(p_img, out_path):
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

   new_im.save(out_path)

if __name__ == "__main__":
   tempDF = pd.read_csv(csv_path)
   #for i in range(10):
   for i in range(len(tempDF["image1"])):
      p_img1 = get_crop_image(i, 1)
      p_img2 = get_crop_image(i, 2)
      p_img3 = get_crop_image(i, 3)
      new_path1 = tempDF["image1"][i].replace("GoogleDataset", "GoogleDataset-new-crop")
      new_path2 = tempDF["image2"][i].replace("GoogleDataset", "GoogleDataset-new-crop")
      new_path3 = tempDF["image3"][i].replace("GoogleDataset", "GoogleDataset-new-crop")
      pad_save_image(p_img1, new_path1)
      pad_save_image(p_img2, new_path2)
      pad_save_image(p_img3, new_path3)
      tempDF["image1"][i] = new_path1
      tempDF["image2"][i] = new_path2
      tempDF["image3"][i] = new_path3
   tempDF.to_csv(save_path, index=False)
