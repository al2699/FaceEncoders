import torch
import torchvision

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
    correct_pred, num_examples = 0, 0
    for i, (x, y) in enumerate(data_loader):

        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        cost = pcc(y_hat, y)
    return cost


