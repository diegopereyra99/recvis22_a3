import argparse
import os
import torch
from torch import optim
from torchvision import datasets
from tqdm import tqdm
from datetime import datetime

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', "-d", type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', "-b", type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', "-e", type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--model', "-m", type=str, default="mobilenet_v2", metavar='M',
                    help='Name of the model to use')                
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
args = parser.parse_args()
args.experiment += "-" + datetime.now().strftime("%d_%m_%H_%M")
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if not os.path.isdir(os.path.join("models", args.experiment)):
    os.makedirs(os.path.join("models", args.experiment))



# Data initialization and loading
from data import train_transforms, val_transforms

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=train_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=val_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)



# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()
model = torch.hub.load('pytorch/vision:v0.10.0', args.model, weights="IMAGENET1K_V2")
model.classifier[-1] = torch.nn.Linear(1280,20)
# torch.save(model, "/tmp/model_tmp.pth")
# model = torch.load("/tmp/model_tmp.pth")

if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.Adam(model.parameters(), lr=args.lr)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=os.path.join("logs", args.experiment)) 

best_val_loss = 1e10
best_val_acc = 0

def train(epoch):
    model.train()
    train_loss = 0
    print(f"\nEpoch {epoch}:")
    for data, target in tqdm(train_loader, total=len(train_loader)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # train loss estimation
        train_loss = train_loss * 0.3 + loss.data.item() * 0.7
    
    print(f"Train set - Average loss: {train_loss:.5f}")
    writer.add_scalars("Loss", {"train": train_loss}, epoch)

def validation(epoch):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader)
    val_acc = correct / len(val_loader.dataset)
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0%})'.format(
        validation_loss, correct, len(val_loader.dataset), val_acc))
    writer.add_scalars("Loss", {"val": validation_loss}, epoch)
    writer.add_scalars("Accuracy", {"val": val_acc}, epoch)
    
    if validation_loss < best_val_loss:
        model_file = "models/" + args.experiment + '/model_min_loss.pth'
        torch.save(model.state_dict(), model_file)

    if val_acc > best_val_acc:
        model_file = "models/" + args.experiment + '/model_max_acc.pth'
        torch.save(model.state_dict(), model_file)
    
for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation(epoch)
    # model_file = "models/" + args.experiment + '/model_' + str(epoch) + '.pth'
    # torch.save(model.state_dict(), model_file)
