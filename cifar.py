import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.misc import face

print("Changed")

transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
        )
trainset = torchvision.datasets.CIFAR10(
        root='./data',train=True,download=True, transform=transform)
trainloader =torch.utils.data.DataLoader(
        trainset, batch_size=64,shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(
        root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
        testset, batch_size=64,shuffle=False,num_workers=4)

classes = ('plane','car','bird','cat','deer','dog',
        'frog','horse','ship','truck')

#---Class Definiition-----------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv4 = nn.Conv2d(192, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*8*256, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        self.gradHist =[]
        self.step=0.01
        self.counter=0
        self.maxSparse = 0.5
        self.Dropout = nn.Dropout(0.25)# Not sure about this one 

    def forward(self, x):
        self.counter +=self.step
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.Dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.Dropout(self.pool(x))
        x = x.view(-1,8*8*256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        #self.gradHist.append(self.fc3.weight.grad)
        return x

def testNet(net,testSet,device):
    net.eval()
    correct = 0
    total = 0
    for data in testSet:
        images, labels = data
        #tensor_image = images.view(images.shape[2],images.shape[0],images.shape[1])
        #tensor_image = tensor_image.view(tensor_image.shape[1],tensor_image.shape[2], tensor_image.shape[0])
        images, labels = Variable(images).to(device), Variable(labels).to(device)
        output = net(images)
        _, predicted = torch.max(output.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    finalAcc =100*correct/total
    print("Accuracy of the network is : {}".format(finalAcc))
    return finalAcc


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
amnt = 0;
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
net.to(device)
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    amnt = int(epoch/2)
    #print(list(net.conv1.named_buffers()))
    for i, data in enumerate(trainloader, 0):#For each batch 
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).cuda()
        

        # reset the parameters to zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print("Gradients size : {}".format(net.conv1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print("Running loss on epoch {} is : {}".format(i,running_loss))
        #if i % 2000 == 1999:    # print every 2000 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0
    print("Loss in epoch {}, is : {}".format(epoch,running_loss/50000))

# We need to verify this b
testNet(net,testloader,device)
#plt.plot(running_loss)
#plt.show()
print("This is the resulting thing : {}".format(net.conv1.named_parameters()))
print("This is are the weights")
prune.l1_unstructured(
        module=net.conv1,
        name="weight",amount=0.5)
#print(net.gradHist)
print(net.conv1.weight)

print('Finished Training')

