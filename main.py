import torch
import torch.nn as nn
import torch.optim as optim

def main():
   model = model()
   model = init_weights(model)
   loss_func = nn.MSELoss()
   #Could later use adam
   optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
   
   epochs = 2001
   

#Special case: only init weights which are on the last fc since
#we want the rest of the restnet weights to be the same
def init_weights(model):
   model.fc.weight.data.fill_(0.01)
   return model

if __name__ == "__main__":
   main()   
