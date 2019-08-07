import torceh
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as f

class Model(nn.Module):
   def __init__(self, vars):
      super(Model, self).__init__()
      
      #Import resnet
      self.rnet = models.resnet50(pretrained=True)
      self.rnet.fc = nn.Linear(in_features=2048, out_features=512)
      self.fc2 = nn.Linear(in_features=512, out_features=16)
      self.norm = f.normalize

   def forward(self, x):
      x = self.rnet(x)
      x = self.fc2(x)
      x = self.norm(x, p=2, dim=-1)
      return x
