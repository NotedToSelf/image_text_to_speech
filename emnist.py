#Erik Bogeberg
#Convolution net to classify EMNIST dataset

import numpy as np
import cv2
import torch
import torchvision
import torchvision.datasets
import matplotlib.pyplot
import tqdm
import os.path
import scipy.io


#
dblTrain = []
dblValidation = []
best_acc = torch.FloatTensor([0])
start_epoch = 0
resume = './emnist/checkpoint.pth.tar'


Train = torch.utils.data.DataLoader(
	batch_size=64,
	shuffle=False,
	num_workers=1,
	pin_memory=False,
	dataset=torchvision.datasets.EMNIST(
		root='./emnist/',
		split='letters',
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(tuple([ 0.1307 ]), tuple([ 0.3081 ]))
		])
	)
)


Validate = torch.utils.data.DataLoader(
	batch_size=64,
	shuffle=True,
	num_workers=1,
	pin_memory=False,
	dataset=torchvision.datasets.EMNIST(
		root='./emnist/',
		split='letters',
		train=False,
		download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor()
		])
	)
)


ObjectOptimizer = torch.optim.Adam(params=moduleNetwork.parameters(), lr=0.001)

#Define the network
class Network(torch.nn.module):
	def __init__(self):
		super(Network, self).__init__()
		
		self.bn = torch.nn.Batchnorm2d(1)
		self.conv1 = torch.nn.conv2d(1, 64, kernel_size = 5)
		self.conv2 = torch.nn.conv2d(64, 512, kernel_size = 5)
		self.linear1(2048, 256)
		self.linear2(256, 128)
		self.linear2(128, 26)

	def forward(self, x):
		x = self.bn(x)
		x = self.conv1(x)	#-1, 64, 24, 24
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.max_pool2d(x, kernel_size = 3) #-1, 64, 8, 8
		x = self.conv2(x)	#-1, 512, 4, 4
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.max_pool2d(x, kernel_size = 2) #-1, 512, 2, 2
		x = view(-1, 2048) #-1, 2048
		x = linear1(x) #-1. 256
		x = torch.nn.functional.dropout(x, p=0.35, training=self.training)
		x = torch.nn.functional.relu(x)
		x = self.linear2(x) #-1, 128
		x = torch.nn.functional.droupout(x, p=0.32, training=self.training)
		x = torch.nn.functional.relu(x)
		x = self.linear3(x) #-1, 26

		return torch.nn.functional.log_softmax(x, dim=1)
		

#If a checkpoint is stored, resume from there
if os.path.isfile(resume):
	checkpoint = torch.load(resume)
	start_epoch = checkpoint['epoch']
	best_acc = checkpoint['best_accuracy']
	emnistNetwork.load_state_dict(checkpoint['state_dict'])


def train():
	#set network to the training mode
	emnistNetwork.train()

	for tensorInput, tensorTarget in tqdm.tqdm(Train):
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=False)#.cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=False)#.cuda()
		objectOptimizer.zero_grad()
		variableEstimate = moduleNetwork(variableInput)
		variableLoss = torch.nn.functional.nll_loss(input=variableEstimate, target=variableTarget)
		variableLoss.backward()
		objectOptimizer.step()


def evaluate():
	#set network to eval mode
	emnistNetwork.eval()

	intTrain = 0
	intValidation = 0
	for tensorInput, tensorTarget in Train:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True)#.cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)#.cuda()
		variableEstimate = moduleNetwork(variableInput)
		intTrain += variableEstimate.data.max(dim=1,keepdim=False)[1].eq(variableTarget.data).sum()

	for tensorInput, tensorTarget in Validate:
		variableInput = torch.autograd.Variable(data=tensorInput,volatile=True)#.cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)#.cuda()
		intValidation += variableEstimate.data.max(dim=1,keepdim=False)[1].eq(variableTarget.data).sum()

		dblTrain.append(100.0 * intTrain / len(Train.dataset))
		dblValidation.append(100.0 * intValidation / len(Validate.dataset))

		print('')
		print('train: ' + str(intTrain) + '/' + str(len(Train.dataset))+ ' (' + str(dblTrain[-1]) + '%)')
		print('validation: ' + str(intValidation) + '/' + str(len(Validate.dataset)) + ' (' + str(dblValidation[-1]) + '%)')
		print('')
		return dblValidation[-1]



#blog.floydhu.com/checkpointing-tutorial-for-tensorflow-keras-and-pytorch/
#credits for checkpointing
def checkpoint(state, best, filename='/emnist/checkpoint.pth.tar'):
	if best:
		print('Saving a new best')
		torch.save(state, filename)
	else:
		print('Validation accuracy did not improve')

#Train for 100 Epochs
for epoch in range(100):
	print('Epoch ' + str(epoch)+ ':')
	train()
	acc = evaluate()

	best = bool(acc.np() > best_acc.np())
	best_acc = torch.FloatTensor(max(acc.np(), best_acc.np()))
	checkpoint({
		'epoch': start_epoch + epoch + 1,
		'state_dict' : emnistNetwork.state_dict(),
		'best_accuracy': best_acc
		}, best)

if False:
	matplotlib.pyplot.figure(figsize=(8.0, 5.0), dpi=150.0)
	matplotlib.pyplot.ylim(79.5, 100.5)
	matplotlib.pyplot.plot(dblTrain)
	matplotlib.pyplot.plot(dblValidation)
	matplotlib.pyplot.show()
