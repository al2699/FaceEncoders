import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import shutil
import data
import random
import cv2

#CSV paths
bp4d_train = "/data1/Alan/BP4D/train.csv"
bp4d_test = "/data1/Alan/BP4D/test.csv"
bp4d_valid = "/data1/Alan/BP4D/valid.csv"

model_save_path = "/home/ICT2000/ahernandez/FaceEncoders/model_finetuned_newN.pt"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#Special case: only init weights which are on the last fc since
#we want the rest of the restnet weights to be the same
def init_weights(model):
   model.fc.weight.data.fill_(0.01)
   return model

def pcc(output, target):
   x = output
   y = target

   vx = (x - torch.mean(x)).double()
   vy = (y - torch.mean(y)).double()

   cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
   return cost

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "/home/ICT2000/ahernandez/FaceEncoders/"
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth')

def validate(model, data_loader, device=None):
    agg_cost = 0
    for i, (x, y) in enumerate(data_loader):
            
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        c = pcc(y_hat, y)
        agg_cost += c
        #print("batch " + str(i) + "'s pcc: " + str(c))
    return agg_cost / len(data_loader)
   
def main():
   model = models.resnet50(pretrained=True)
   #for param in model.parameters():
      #param.requires_grad=False
   model.fc = nn.Linear(in_features=2048, out_features=5)
   model = init_weights(model)
   model.train()
   #loss_func = nn.MSELoss()
   #loss_func = pcc
   loss_func = nn.MSELoss()
   #Could later use adam
   optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.001)
   print("Cuda available?: " + str(torch.cuda.is_available()))
   model = model.to(device)

   custom_transform = transforms.Compose([transforms.ToTensor(),
                                       #transforms.ToPILImage(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

   #Load datasets
   print("Loading data...")
   train = data.BP4DDataset(csv_file=bp4d_train, transform=custom_transform)
   test = data.BP4DDataset(csv_file=bp4d_test, transform=custom_transform)
   valid = data.BP4DDataset(csv_file=bp4d_valid, transform=custom_transform)
   #TODO: Number of workers
   train_dl = DataLoader(train,
                         batch_size=128,
                         shuffle=True,
                         num_workers=4)
   test_dl = DataLoader(test, 
                        batch_size=128,
                        shuffle=True,
                        num_workers=4)
   valid_dl = DataLoader(valid, 
                         batch_size=128,
                         shuffle=True,
                         num_workers=4)
   print("Data loaded")

   #could improve upon this by using data loaders for mini-batch sampling
   epochs = 1000
   
   print("Beginning training...")
   best_acc = -100000
   for epoch in range(epochs):
      #train on 300W dataset
      model.train()
      #model.eval()
      #validate(model, valid_dl, device=device)
      for batch_idx, (x_var, y_var) in enumerate(train_dl):
         x_var = x_var.to(device)
         y_var = y_var.to(device)
         #forward/backprop
         y_hat = model(x_var)
         #print("y_hat: " + str(y_hat))
         #print("y_hat len: " + str(len(y_hat[0])))
         #print("y len: " + str(len(y_var[0])))
         #input("Press enter to continue")
         cost = loss_func.forward(y_hat, y_var.float())
         optimizer.zero_grad()

         #Update model parameters
         cost.backward()

         #Update model parameters
         optimizer.step()
         #Logging
         if not batch_idx % 50:
             print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, epochs, batch_idx, 
                      len(train_dl), cost))
      model.eval()
      with torch.set_grad_enabled(False): # save memory during inference
        acc = validate(model, valid_dl, device=device)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
        }, is_best)
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            epoch+1, epochs, 
             validate(model, train_dl, device=device),
             acc))
  
   
   #save model weights
   print("Saving model to: "+ model_save_path)
   torch.save(model.state_dict(), model_save_path)
      
if __name__ == "__main__":
   main()
