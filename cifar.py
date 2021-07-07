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

### Macros ###
finalSparsityLevel = 0.4
epochs = 2
toPrune = False
### Macros ###

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

    def forward(self, x_i):
        self.counter +=self.step
        x = F.relu(self.conv1(x_i))
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
        # Also no need to one hot encode because we are comparing indexes
        _, predicted = torch.max(output.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    finalAcc =100*correct/total
    print("Accuracy of the network is : {}".format(finalAcc))
    return finalAcc

def pruneLayers(net,pruneLevel):
    prune.l1_unstructured(
        module=net.conv1,
        name="weight", amount=pruneLevel)
    prune.l1_unstructured(
        module=net.conv2, name="weight", amount=pruneLevel)
    prune.l1_unstructured(
        module=net.conv3, name="weight", amount=pruneLevel)
    prune.l1_unstructured(
        module=net.conv4, name="weight", amount=pruneLevel)
    returno =[1 - (torch.count_nonzero(net.conv1.weight) / torch.numel(net.conv1.weight)),
              (1- torch.count_nonzero(net.conv2.weight) / torch.numel(net.conv2.weight)),
              (1- torch.count_nonzero(net.conv3.weight) / torch.numel(net.conv3.weight)),
              (1- torch.count_nonzero(net.conv4.weight) / torch.numel(net.conv4.weight))]
    print("This are the stats : ")
    print(returno)
    return returno

def extractVOG(net,testSet,device):
    # Lets do the VOG thing here
    print("Evaluating VOG")
    counter = 0
    net.eval()  # Turn training
    grads = []
    for images,labels in testSet:# Per batch
        images, labels = Variable(images).to(device), Variable(labels).to(device)
        labelsShape = labels.shape
        ones  = torch.ones(labelsShape)
        #images.requires_grad = True
        #input = transform(images[0])
        input = images[0]
        input.unsqueeze_(0)
        input.requires_grad=True
        #logits = net(images)
        preds = net(input)
        #layer_softMax = torch.nn.Softmax(dim=1)(logits)
        score, indices = torch.max(preds,1)
        # the tensor.type below casts the tensor into a LongTensor type(rather than float i think)
        #sel_nodes = layer_softMax[torch,torch.arange(len(labels),labels.type(torch.LongTensor))]# I dont understand much why this is happening
#        layer_softMax.backward(ones)
        score.backward()
        grads.append(input.grad.cpu().data.numpy())
        counter += 1
        if counter > 1:
            break
    print("Finished VOG Evaluation")
    return grads

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
amnt = 0;
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print("Device is : {}".format(device))
net.to(device)

curSparsityVal = 0.0
previousSparsityLevel = 0.0
pruneRes = []
lossPerEp = []

for epoch in range(epochs):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0
    amnt = int(epoch/2)
    curSparsityVal = finalSparsityLevel - finalSparsityLevel*pow(1-(epoch/epochs),3)
    print("Current Sparsity level = {}".format(curSparsityVal))
    print("On epoch : {}, amount of sparsifying is : {}".format(epoch,curSparsityVal-previousSparsityLevel))
    #print(list(net.conv1.named_buffers()))
    if toPrune:# Pruning here(if enabled)
        pruneRes.append(pruneLayers(net,curSparsityVal-previousSparsityLevel ))

    # Per batch
    for i, data in enumerate(trainloader, 0):#For each batch
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device), Variable(labels).cuda()


        # reset the parameters to zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print("Gradients size : {}".format(net.conv1))
        # No need to one-hot encode because criterion takes index labels
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
    lossPerEp.append(running_loss/50000)
    previousSparsityLevel = curSparsityVal



print("Now printing results of sparsification with net:")
ax1 = plt.subplot(211,title="Error")
plt.plot(np.arange(0,epochs),lossPerEp,color='red',linestyle='dashed')
plt.subplot(212,sharex=ax1, title="Pruning Stats")
pruneRes = np.array([j.to('cpu') for i in pruneRes for j in i])
pruneRes = np.resize(pruneRes,(epochs,4))
plt.plot(pruneRes)
plt.show()

### Testing Net ####
testNet(net,testloader,device)

### Extracting Gradients ###
grads = extractVOG(net,testloader,device)
ax1 = plt.subplot(211,title="Original Image")
plt.imshow(np.moveaxis(np.asarray(testset[0][0].cpu()),0,-1).astype('uint8'))
ax2 = plt.subplot(212,title="Gradient Image")
plt.imshow(np.moveaxis(grads[0][0],0,-1))

plt.show()

# We need to verify this b
#plt.plot(running_loss)
#plt.show()
print("This is the resulting thing : {}".format(net.conv1.named_parameters()))
print("This is are the weights")
#print(net.gradHist)
print(net.conv1.weight)

print('Finished Training')
