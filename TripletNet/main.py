#!/home/ICT2000/ahernandez/anaconda3/envs/myenv/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import data
import random
import cv2

#CSV paths
fec_train = "/data1/Alan/GoogleDataset/train.csv"
fec_test = "/data1/Alan/GoogleDataset/fec_test_new1.csv"
fec_valid = "/data1/Alan/GoogleDataset/valid.csv"


#TODO: change N to num of epochs
model_save_path = "/data2/Alan/FaceEncoders/TripletNet/triplet_finetuned_Ne.pt"
#TODO: Change to cuda:0 when on 
#Special case: only init weights which are on the last fc since
#we want the rest of the restnet weights to be the same
def init_weights(model):
   model.fc.weight.data.fill_(0.01)
   return model

#ASSSUMING img1Embed and img2Embed are most similar img embeddings
def triplet_loss(img1Embed, img2Embed, img3Embed, margin, device=None):
   #Let sim_pair := e1 - e2; not_sim_pair1 := e1 - e3; not_sim_pair2 := e2 - e3
   similar_pair_diff = torch.sub(img1Embed, img2Embed)
   #might be better to flatten out to 1 and take regular euclidean norm
   sim_pair_norm = torch.norm(similar_pair_diff, p=2, dim=-1)
   #should be 40 x 1
   sim_pair_ns = torch.mul(sim_pair_norm, sim_pair_norm)
   not_sim_diff1 = torch.sub(img1Embed,img3Embed)
   not_sim_norm1 = torch.norm(not_sim_diff1, p=2, dim=-1)
   not_sim_ns1 = torch.mul(not_sim_norm1, not_sim_norm1)
   not_sim_diff2 = torch.sub(img2Embed, img3Embed)
   not_sim_norm2 = torch.norm(not_sim_diff2, p=2, dim=-1)
   not_sim_ns2 = torch.mul(not_sim_norm2, not_sim_norm2)
   
   #intermediate steps
   diff1 = torch.sub(sim_pair_ns, not_sim_ns1)
   f_add1 = torch.add(diff1.double().to(device), margin.double().to(device))
   diff2 = torch.sub(sim_pair_ns, not_sim_ns2)
   f_add2 = torch.add(diff2.double().to(device), margin.double().to(device))

   #Rest of loss
   zero_tensor = torch.Tensor([0])
   zero_var = Variable(zero_tensor, requires_grad=False)
   zero_var = zero_var.to(device)
   f = torch.max(zero_var, f_add1.float())
   s = torch.max(zero_var, f_add2.float())
   loss = torch.add(f, s)
   
   return loss.mean()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    directory = "/data2/Alan/FaceEncoders/TripletNet/"
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth')

def validate(model, data_loader, device=None):
   num_correct = 0
   num_examples = 0
   cos_sim = nn.CosineSimilarity(dim=16)
   one_tensor = torch.Tensor([1])
   one_var = Variable(one_tensor, requires_grad=False)
   one_var.to(device)

   for i, (img1, img2, img3, margin) in enumerate(data_loader):

      img1 = img1.to(device)
      e1 = model(img1)
      img2 = img2.to(device)
      e2 = model(img2)
      img3 = img3.to(device)
      e3 = model(img3)

      #Use cosine distance to find embedding distances
      sim_dist = torch.sub(one_var, cos_sim.forward(e1, e2))
      diff_dist1 = torch.sub(one_var, cos_sim.forward(e2, e3))
      diff_dist2 = torch.sub(one_var, cos_sim.forward(e1, e3))

      num_examples += img1.size(0)
      num_correct += (sim_dist < diff_dist1 and sim_dist < diff_dist2).sum()
   prediction_accuracy = num_correct / num_examples
   return prediction_accuracy


def main():
   #For model
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   #For images
   #import resnet 50 layers to fine tune
   model = models.resnet50(pretrained=True)
   #Output to 16-d embedding
   model.fc = nn.Linear(in_features=2048, out_features=16)
   model = init_weights(model)
   print("Cuda available?: " + str(torch.cuda.is_available()))
   model.to(device)
   #input("Model.to(device)")
   #Could later use adam
   #Loss func goes here
   optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.01)


   custom_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
   #Load datasets
   print("Loading data...")
   train = data.FECDataset(csv_file=fec_train, transform=custom_transform)
   test = data.FECDataset(csv_file=fec_test, transform=custom_transform)
   valid = data.FECDataset(csv_file=fec_valid, transform=custom_transform)

   train_dl = DataLoader(train,
                         batch_size=40,
                         shuffle=True,
                         num_workers=4)
   test_dl = DataLoader(test,
                        batch_size=40,
                        shuffle=True,
                        num_workers=4)
   valid_dl = DataLoader(valid,
                         batch_size=40,
                         shuffle=True,
                         num_workers=4)

   print("Data loaded")

   #could improve upon this by using data loaders for mini-batch sampling
   epochs = 500
   
   print("Beginning training...")
   for epoch in range(epochs):
      #input("About to start training")
      model.train()
      for batch_idx, (img1, img2, img3, margin) in enumerate(train_dl):
         img1 = img1.to(device)
         img2 = img2.to(device)
         img3 = img3.to(device)

         #forward/backprop
         #input("Added imgs to devices")
         e1 = model(img1)
         #input("after model(img1)")
         e2 = model(img2)
         #input("after model(img2)")
         e3 = model(img3)
         #input("after model(img3)")
         #print("y_hat: " + str(y_hat))
         #print("y_hat len: " + str(len(y_hat[0])))
         #print("y len: " + str(len(y_var[0])))
         #input("After model(img1)")
         cost = triplet_loss(e1, e2, e3, margin,
                                 device=device)
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
      #Begin inference
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
