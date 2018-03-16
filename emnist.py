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




#SOURCE for converting from MAT to Python format: github.com/j05t/emnist/blob/master/emnist.ipynbdddd


#read in MAT format dataset
emnist = scipy.io.loadmat('./emnist/data/letters.mat')

#extract data
training_data = emnist["dataset"][0][0][0][0][0][0]
training_data = training_data.astype(np.float32)
training_labels = emnist["dataset"][0][0][0][0][0][1]

test_data = emnist["dataset"][0][0][1][0][0][0]
test_data = test_data.astype(np.float32)
test_labels = emnist["dataset"][0][0][1][0][0][1]

#normalize
training_data /= 255
test_data /= 255

#reshape from matlab order to python order
training_data = training_data.reshape(training_data.shape[0], 1, 28, 28, order="A")
test_data = test_data.reshape(test_data.shape[0], 1, 28, 28, order="A")


#Some tests that data is loaded correctly:
if False:
	print('training data: ' + str(training_data.shape))
	print('training labels: ' + str(training_labels.shape))
	print('test data: ' + str(test_data.shape))
	print('test labels: ' + str(test_labels.shape))
	test_im = training_data[1277]
	print(test_labels[1277][0])
	cv2.imwrite(filename='./test.png', img=(test_im[0] * 255.0).clip(0.0, 255.0).astype(np.uint8))



#Define the Network:
class Network(torch.nn.Module):

	def __init__(self):
		super(Network, self).__init__()

		self.conv1 = torch.nn.Conv2d(1,32, kernel_size=5)
		self.conv2 = torch.nn.Conv2d(32,64, kernel_size=5)
		self.fc1 = torch.nn.Linear(256, 200)
		self.fc2 = torch.nn.Linear(200,10)

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



emnistNetwork = Network().cuda()

objectOptimizer = torch.optim.Adam(params=emnistNetwork.parameters(), lr=0.001)


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
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=False).cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=False).cuda()
		
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
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True).cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True).cuda()
		variableEstimate = emnistNetwork(variableInput)

		intTrain += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()

	for tensorInput, tensorTarget in Validation:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True).cuda()
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True).cuda()
		variableEstimate = emnistNetwork(variableInput)
		
		intValidation += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()

	#Determine accuracy
	dblTrain.append(100.0 * intTrain / len(Train.dataset))
	dblValidation.append(100.0 * intValidation / len(Validation.dataset))

	#Display data
	print('')
	print('train: ' + str(intTrain) + '/' + str(len(Train.dataset)) + ' (' + str(dblTrain[-1]) + '%)')
	print('validation: ' + str(intValidation) + '/' + str(len(Validation.dataset)) + ' (' + str(dblValidation[-1]) + '%)')
	print('')

	#return validation percent for checkpointing
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
