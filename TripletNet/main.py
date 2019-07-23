import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
#import data
import random
import cv2

model_save_path = "/home/ICT2000/ahernandez/Documents/FaceEncoders/"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#Special case: only init weights which are on the last fc since
#we want the rest of the restnet weights to be the same
def init_weights(model):
   model.fc.weight.data.fill_(0.01)
   return model

#Assuming img1Embed and img2Embed are most similar img embeddings
def triplet_loss(img1Embed, img2Embed, img3Embed, margin):
   #Let sim_pair := e1 - e2; not_sim_pair1 := e1 - e3; not_sim_pair2 := e2 - e3
   similar_pair_diff = torch.sub(img1Embed, img2Embed)
   #might be better to flatten out to 1 and take regular euclidean norm
   sim_pair_norm = torch.norm(similar_pair_diff, p=2, dim=16)
   sim_pair_ns = torch.mul(sim_pair, sim_pair)
   not_sim_diff1 = torch.sub(img1Embed,img3Embed)
   not_sim_norm1 = torch.norm(not_sim_diff1, p=2, dim=16)
   not_sim_ns1 = torch.mul(not_sim_norm1, not_sim_norm1)
   not_sim_diff2 = torch.sub(img2Embed, img3Embed)
   not_sim_norm2 = torch.norm(not_sim_diff2, p=2, dim=16)
   not_sim_ns2 = torch.mul(not_sim_norm2, not_sim_norm2)
   
   #intermediate steps
   diff1 = torch.sub(sim_pair_ns, not_sim_ns1)
   f_add1 = torch.add(diff1, margin)
   diff2 = torch.sub(sim_pair_ns, not_sim_ns2)
   f_add2 = torch.add(diff2, margin)

   #Rest of loss
   loss = torch.add(torch.max(0, f_add1), torch.max(0, f_add2))
   return loss

def main():
   model = models.resnet50(pretrained=True)
   model.fc = nn.Linear(in_features=2048, out_features=16)
   model = init_weights(model)
   model.train()
   #loss_func = 
   #Could later use adam
   optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.2)
   print("Cuda available?: " + str(torch.cuda.is_available()))
   model = model.to(device)

   #Load datasets
   print("Loading data...")
   w300 = data.W300Dataset()
   w300_train_ind, w300_test_ind, w300_validate_ind = w300.train_test_validation_split()
   print("Data loaded")

   #could improve upon this by using data loaders for mini-batch sampling
   epochs = 2001
   
   print("Beginning training...")
   for i in range(epochs):
      print("epoch: " + str(i))
      #train on 300W dataset
      model.train()
      for j in range(len(w300_train_ind)):
         #print("Step: " + str(j))
         #300W dataset train step
         x_var, y_var = w300[w300_train_ind[j]]
         y_var = y_var.to(device)
         #temp addition
         x_var = x_var.unsqueeze(0)
         x_var = x_var.view(1, 3, 224, 224)
         
         optimizer.zero_grad()
         x_var = x_var.to(device)
         y_hat = model(x_var).view(-1, 1)
         #print("y_var: " + str(y_var))
         #print("y_hat: " + str(y_hat))
         loss = loss_func.forward(y_hat, y_var)
         loss.backward()
         optimizer.step()
         #print("x_var: " + str(x_var))
         #print("y_hat: " + str(y_var))
    
      for j in range(len(ck_train_ind)):     
         #CK+ dataset train step
         x_var, y_var = ck[ck_train_ind[j]]
         y_var = y_var.to(device)
         x_var = x_var.unsqueeze(0)
         x_var = x_var.view(1,3,224,224)

         optimizer.zero_grad()
         x_var = x_var.to(device)
         y_hat = model(x_var).view(-1,1)
         #print("y_var: " + str(y_var))
         #print("y_hat: " + str(y_hat))
         loss = loss_func.forward(y_hat, y_var)
         loss.backward()
         optimizer.step()
         #print("x_var: " + str(x_var))
         #print("y_hat: " + str(y_hat))
      
      for j in range(len(bp4d_train_ind)):
         #BP4D dataset train step
         x_var, y_var = bp4d[bp4d_train_ind[j]]
         y_var = y_var.to(device)
         #temp addition
         x_var = x_var.unsqueeze(0)
         x_var = x_var.view(1, 3, 224, 224)

         optimizer.zero_grad()
         x_var = x_var.to(device)
         y_hat = model(x_var).view(-1, 1)
         #print("y_var: " + str(y_var))
         #print("y_hat: " + str(y_hat))
         loss = loss_func.forward(y_hat, y_var)
         loss.backward()
         optimizer.step()
         #print("x_var: " + str(x_var))
         #print("y_hat: " + str(y_hat))
      
      if(i % 10 == 0):
         print("300W validation:")
         validate(w300_validate_ind, w300, model, loss_func)
         print("CK+ validation:")
         validate(ck_validate_ind, ck, model, loss_func)
         print("BP4D validation:")
         validate(bp4d_validate_ind, bp4d, model, loss_func)
         #input("waiting for key press")
  
   print("Test/final loss: ")
   print("300W test:")
   validate(w300_test_ind, w300, model, loss_func)
   print("CK+ test")
   validate(ck_test_ind, ck, model, loss_func)
   
   #save model weights
   print("Saving model to: "+ model_save_path)
   torch.save(model, model_save_path)


def validate(valid_indices, dataset, model, loss_func):
   agg_loss = 0
   model.eval()
   for i in range(len(valid_indices)):
      #print("Validating: " + str(type(dataset)))
      #print("Grabbed index: " + str(valid_indices[i]) + " Len: " + str(len(valid_indices)))
      x_var, y_var = (None,None) #initialize for interpreter
      try:
         x_var, y_var = dataset[int(valid_indices[i])]
      except:
         break  
      x_var = x_var.unsqueeze(0)
      x_var = x_var.view(1, 3, 224, 224)
      x_var = x_var.to(device)
      y_var = y_var.to(device)
      y_hat = model(x_var).view(-1,1)
      loss = loss_func.forward(y_hat, y_var)
      #print("x_var: " + str(x_var))
      #print("y_hat: " + str(y_hat))
      if i == 5: print("Random loss: " + str(loss.data))
      agg_loss += loss.item()
   print("Aggregate loss: " + str(agg_loss))
   print("Average loss: " + str(agg_loss / len(valid_indices)))

if __name__ == "__main__":
   main()   
