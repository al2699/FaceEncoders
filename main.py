import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import data

model_save_path = "/home/ICT2000/ahernandez/Documents/FaceEncoders/"

#Special case: only init weights which are on the last fc since
#we want the rest of the restnet weights to be the same
def init_weights(model):
   model.fc.weight.data.fill_(0.01)
   return model

def main():
   model = models.resnet50(pretrained=True)
   model.fc = nn.Linear(in_features=2048, out_features=41)
   model = init_weights(model)
   model.train()
   loss_func = nn.MSELoss()
   #Could later use adam
   optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   model = model.to(device)

   #Load datasets
   w300 = W300Dataset(os.getcwd())
   w300_train_ind, w300_test_ind, w300_validate_ind = train_test_validate(w300)
   ck = CKDataset(os.getcwd())
   ck_train_ind, ck_test_ind, ck_validate_ind = train_test_validate(ck)
   #BP4D = BP4DDataset(os.getcwd())

   #could improve upon this by using data loaders for mini-batch sampling
   epochs = 2001
   steps = len(W300) + len(Ck) #+ len(BP4D)
   
   for i in range(epochs):
      #train on 300W dataset
      model.train()
      for j in range(len(w300_train_ind)):
         #300W dataset train step
         x_var, y_var = w300[w300_train_ind[j]]
         
         optimizer.zero_grad()
         y_hat = model(x_var)
         loss = loss_func.forward(y_hat, y_var)
         loss.backwards()
         optimizer.step()
         
         #CK+ dataset train step
         x_var, y_var = ck[ck_train_ind[j]]
         
         optimizer.zero_grad()
         y_hat = model(x_var)
         loss = loss_func.forward(y_hat, y_var)
         loss.backwards()
         optimizer.step()
         
      if(i % 500 == 0):
         print("300W validation:")
         validate(w300_validate_ind, w300, model, loss_func)
         print("CK+ validation")
         validate(ck_validate_ind, ck, model, loss_func)
  
   print("Test/final loss: ")
   print("300W test:")
   validate(w300_test_ind, w300, model, loss_func)
   print("CK+ test")
   validate(ck_test_ind, ck, model, loss_func)
   
   #save model weights
   print("Saving model to: "+ model_save_path)
   torch.save(model, model_save_path)

def train_test_validate_split(dataset):
   arr = range(0, len(dataset))
   #10% valid, %10 testing, 80% training
   train_amount = int(.10 * len(dataset))
   test_amount = int(.10 * len(dataset))
   train_indices = []
   test_indices = []

   for i in range(test_amount):
      pick = random.randint(0, len(arr) - 1)
      test_indices.append(arr[pick])
      del arr[pick]
   train_indices = arr

   return train_indices, test_indices

def validate(valid_indices, dataset, model, loss_func):
   agg_loss = 0
   model.eval()
   for i in range(len(valid_indices)):
      img, label = dataset[valid_indices[i]]
      y_hat = model(img)
      loss = loss_func.forward(y_hat, label)
      agg_loss += loss
   print("Aggregate loss validation: " + str(agg_loss))
   print("Average loss: " + str(agg_loss / len(valid_indices)))
      
if __name__ == "__main__":
   main()   
