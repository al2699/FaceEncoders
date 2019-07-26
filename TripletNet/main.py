import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
import os
import data
import random
import cv2

model_save_path = "/data2/Alan/FaceEncoders/TripletNet/triplet_finetuned.pt"
fec_test_path = "/data1/Alan/GoogleDataset/fec_test_new1.csv" #TODO: FILL THIS IN
#TODO: Change to cuda:0 when on 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Special case: only init weights which are on the last fc since
#we want the rest of the restnet weights to be the same
def init_weights(model):
   model.fc.weight.data.fill_(0.01)
   return model

#ASSSUMING img1Embed and img2Embed are most similar img embeddings
def triplet_loss(img1Embed, img2Embed, img3Embed, margin):
   #Let sim_pair := e1 - e2; not_sim_pair1 := e1 - e3; not_sim_pair2 := e2 - e3
   similar_pair_diff = torch.sub(img1Embed, img2Embed)
   #might be better to flatten out to 1 and take regular euclidean norm
   sim_pair_norm = torch.norm(similar_pair_diff, p=2, dim=-1)
   sim_pair_ns = torch.mul(sim_pair_norm, sim_pair_norm)
   not_sim_diff1 = torch.sub(img1Embed,img3Embed)
   not_sim_norm1 = torch.norm(not_sim_diff1, p=2, dim=-1)
   not_sim_ns1 = torch.mul(not_sim_norm1, not_sim_norm1)
   not_sim_diff2 = torch.sub(img2Embed, img3Embed)
   not_sim_norm2 = torch.norm(not_sim_diff2, p=2, dim=-1)
   not_sim_ns2 = torch.mul(not_sim_norm2, not_sim_norm2)
   
   #intermediate steps
   diff1 = torch.sub(sim_pair_ns, not_sim_ns1)
   f_add1 = torch.add(diff1, margin)
   diff2 = torch.sub(sim_pair_ns, not_sim_ns2)
   f_add2 = torch.add(diff2, margin)

   #Rest of loss
   zero_tensor = torch.Tensor([0])
   zero_var = Variable(zero_tensor, requires_grad=False)
   zero_var = zero_var.to(device)
   f = torch.max(zero_var, f_add1)
   s = torch.max(zero_var, f_add2)
   loss = torch.add(f, s)
   #print("Loss: " + str(loss))
   return loss

def main():
   #import resnet 50 layers to fine tune
   model = models.resnet50(pretrained=True)
   #Output to 16-d embedding
   model.fc = nn.Linear(in_features=2048, out_features=16)
   model = init_weights(model)
   model.train()
   #Could later use adam
   #Loss func goes here
   optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.2)
   print("Cuda available?: " + str(torch.cuda.is_available()))
   model = model.to(device)

   #Load datasets
   print("Loading data...")
   fec = data.FECDataset()
   fec_train_ind, fec_valid_ind = fec.train_valid_split()
   fec_test = data.FECDataset(csv_file=fec_test_path)
   print("Data loaded")

   #could improve upon this by using data loaders for mini-batch sampling
   epochs = 1
   
   print("Beginning training...")
   for i in range(epochs):
      print("epoch: " + str(i))
      #train on FEC dataset
      model.train()
      for j in range(len(fec_train_ind)):
         #Grab random data point in order to avoid temporal benefits
         dp_index = random.randint(0, len(fec_train_ind) - 1)
         print("j: " + str(j))
         #FEC dataset train step
         img1, img2, img3, margin = fec[fec_train_ind[dp_index]]
         img1 = img1.unsqueeze(0)
         img1 = img1.view(1,3,224,224)
         img2 = img2.unsqueeze(0)
         img2 = img2.view(1,3,224,224)
         img3 = img3.unsqueeze(0)
         img3 = img3.view(1,3,224,224)
         
         optimizer.zero_grad()
         img1 = img1.to(device)
         y_hat1 = model(img1)
         img2 = img2.to(device)
         y_hat2 = model(img2)
         img3 = img3.to(device)
         y_hat3 = model(img3)

         loss = triplet_loss(y_hat1, y_hat2, y_hat3, margin)
         loss.backward()
         optimizer.step()
      
      if(i % 10 == 0):
         print("Validation:")
         validate(fec_valid_ind, fec, model)
  
   print("Test/final loss: ")
   test(fec, model)

   #save model weights
   print("Saving model to: "+ model_save_path)
   torch.save(model.state_dict(), model_save_path)


def validate(valid_indices, dataset, model):
   agg_loss = 0
   model.eval()
   validate_size = int(len(dataset) * 0.1)
   for i in range(len(validate_size)):
      #print("Validating: " + str(type(dataset)))
      #print("Grabbed index: " + str(valid_indices[i]) + " Len: " + str(len(valid_indices)))
      img1, img2, img3, margin  = (None, None, None, 0.0) #initialize for interpreter
      try:
         img1, img2, img3, margin  = dataset[i]
      except:
         break
      img1 = img1.unsqueeze(0)
      img1 = img1.view(1, 3, 224, 224)
      img2 = img2.unsqueeze(0)
      img2 = img2.view(1, 3, 224, 224)
      img3 = img3.unsqueeze(0)
      img3 = img3.view(1, 3, 224, 224)

      img1 = img1.to(device)
      y_hat1 = model(img1)
      img2 = img2.to(device)
      y_hat2 = model(img2)
      img3 = img3.to(device)
      y_hat3 = model(img3)

      loss = triplet_loss(y_hat1, y_hat2, y_hat3, margin)
      #print("x_var: " + str(x_var))
      #print("y_hat: " + str(y_hat))
      if i == 5: print("Random loss: " + str(loss.data))
      agg_loss += loss.item()
   print("Aggregate loss: " + str(agg_loss))
   print("Average loss: " + str(agg_loss / len(valid_indices)))

def test(dataset, model):
   agg_loss = 0
   for i in range(len(dataset)):
      img1, img2, img3, margin  = dataset[i]

      img1 = img1.unsqueeze(0)
      img1 = img1.view(1, 3, 224, 224)
      img2 = img2.unsqueeze(0)
      img2 = img2.view(1, 3, 224, 224)
      img3 = img3.unsqueeze(0)
      img3 = img3.view(1, 3, 224, 224)

      img1 = img1.to(device)
      y_hat1 = model(img1)
      img2 = img2.to(device)
      y_hat2 = model(img2)
      img3 = img3.to(device)
      y_hat3 = model(img3)
    
      loss = triplet_loss(y_hat1, y_hat2, y_hat3, margin)
      agg_loss += loss.item()
   print("Agg loss: " + str(agg_loss))
   print("Average loss: " + str(agg_loss/len(dataset)))
if __name__ == "__main__":
   main()
