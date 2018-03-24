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
import sys
import PIL
from PIL import Image

dblTrain = []
dblValidation = []
best_acc = torch.FloatTensor([0])
start_epoch = 0
cwd = os.getcwd()
resume = str(cwd) + '/emnist/checkpoint.pth.tar'
data = str(cwd) + '/data/'
mode = sys.argv[1]
k = 0
assert(mode == 'train' or mode == 'classify')

#Define the network
class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()
	
		self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5)
		self.fc1 = torch.nn.Linear(256, 200)
		self.fc2 = torch.nn.Linear(200, 26)


	def forward(self, x):
		x = self.conv1(x)
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.max_pool2d(x, kernel_size=3)
		x = self.conv2(x)
		x = torch.nn.functional.relu(x)
		x = torch.nn.functional.max_pool2d(x, kernel_size=2)
		x = x.view(-1, 256)
		x = self.fc1(x)
		x = torch.nn.functional.relu(x)
		x = self.fc2(x)

		return torch.nn.functional.log_softmax(x, dim=1)
		

emnistNetwork = Network()

#If a checkpoint is stored, resume from there
if os.path.isfile(resume):
	checkpoint = torch.load(resume)
	start_epoch = checkpoint['epoch']
	best_acc = checkpoint['best_accuracy']
	emnistNetwork.load_state_dict(checkpoint['state_dict'])
	print('\n\nFound a checkpoint. Trained for ' + str(start_epoch) + ' epochs with max ' + str(best_acc[0]) + ' accuracy\n')
else:
	print('\n\nNo checkpoint Stored\n')



objectOptimizer = torch.optim.Adam(params=emnistNetwork.parameters(), lr=0.001)


if mode == 'train':

	print('\nStarting in training mode. . .')

	Train = torch.utils.data.DataLoader(
		batch_size=64,
		shuffle=True,
		num_workers=1,
		pin_memory=False,
		dataset=torchvision.datasets.EMNIST(
			root='./emnist/',
			split='letters',
			train=True,
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
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(tuple([ 0.1307 ]), tuple([ 0.3081 ]))
			])
		)
	)




	def train():
		#set network to the training mode
		emnistNetwork.train()

		for tensorInput, tensorTarget in tqdm.tqdm(Train):
			variableInput = torch.autograd.Variable(data=tensorInput, volatile=False)
			variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=False)
			variableTarget.data = variableTarget.data-1
			objectOptimizer.zero_grad()

			variableEstimate = emnistNetwork(variableInput)
			variableLoss = torch.nn.functional.nll_loss(input=variableEstimate, target=variableTarget)
			variableLoss.backward()
			objectOptimizer.step()


	def evaluate():
		#set network to eval mode
		emnistNetwork.eval()

		intTrain = 0
		intValidation = 0
		for tensorInput, tensorTarget in Train:
			variableInput = torch.autograd.Variable(data=tensorInput, volatile=True)
			variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)
			variableTarget.data = variableTarget.data-1

			variableEstimate = emnistNetwork(variableInput)
			intTrain += variableEstimate.data.max(dim=1,keepdim=False)[1].eq(variableTarget.data).sum()

		for tensorInput, tensorTarget in Validate:
			variableInput = torch.autograd.Variable(data=tensorInput,volatile=True)
			variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)
			variableTarget.data = variableTarget.data-1
			variableEstimate = emnistNetwork(variableInput)	
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
	def checkpoint(state, best, filename=resume):
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

		best = bool(float(acc) > float(best_acc))
		num = (max(float(acc), float(best_acc)))
		best_acc = torch.FloatTensor([num])

		checkpoint({
			'epoch': start_epoch + epoch + 1,
			'state_dict' : emnistNetwork.state_dict(),
			'best_accuracy': best_acc
			}, best)

if mode == 'classify':
	
	print('\nStarting in classification mode . . .')
	print('\nLoading data')


	#Image loader function which reads single channel images
	def image_loader(path):
		with open(path, 'rb') as f:
			img = Image.open(f)
			img = PIL.ImageOps.grayscale(img)
			return img

	#Create data loader for image folder	
	objectData = torch.utils.data.DataLoader(
		batch_size=1,
		shuffle=False,
		num_workers=1,
		pin_memory=False,
		dataset=torchvision.datasets.ImageFolder(
			root=data,
			transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor()
				#torchvision.transforms.Normalize(tuple([ 0.1307 ]), tuple([ 0.3081 ]))
			]),
		loader = image_loader
		)
	)


	def map_to_char(num):
		return {
			0 : 'a',
			1 : 'b',
			2 : 'c',
			3 : 'd',
			4 : 'e',
			5 : 'f',
			6 : 'g',
			7 : 'h',
			8 : 'i',
			9 : 'j',
			10: 'k',
			11: 'l',
			12: 'm',
			13: 'n',
			14: 'o',
			15: 'p',
			16: 'q',
			17: 'r',
			18: 's',
			19: 't',
			20: 'u',
			21: 'v',
			22: 'w',
			23: 'x',
			24: 'y',
			25: 'z'
		}[num]

	chars = []
	k = 0
	print('\nClassifying data')
	for tensorInput in objectData:
		image = tensorInput[0]
		inp = image[0][0]
		inp = inp.unsqueeze(0)
		inp = inp.unsqueeze(0)
		inp = torch.autograd.Variable(data=inp)
		estimate = emnistNetwork(inp)
		num, ind = torch.max(estimate, 1)
		chars.append(map_to_char(int(ind[0])))
		
	print('\nWriting data')
	f = open("output.txt", "w+")
	for i in range(len(chars)):
		f.write(chars[i] + ' ')
