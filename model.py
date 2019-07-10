import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models



class Model(nn.Module):
   def __init__(self, vars):
      super(Model, self).__init__()
      
      #Import resnet
      self.model = models.resnet50(pretrained=True)
      self.model.fc = nn.Linear(in_features=2048, out_features=41)

   def forward(self, x):
      return self.model(x)
