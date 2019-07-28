import torchvision.transforms as t
from PIL import Image
import pandas as pd

#Helper function
def get_pil_image(img_path):
   toTensor = t.ToTensor()
   toPImage = t.ToPILImage()
   return toPImage(toTensor(img_path))

def pad_iamge(img_path, side):
   p_img = get_pil_image(img_path
)
   if side == "height":
      
   else:

csv_path = ""

df = pd.read_csv(csv_path)


