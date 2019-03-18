
import torch
import json
import pickle
import numpy as np
from torchvision import datasets
from torchvision import transforms



class Preprocess:

	def __init__(self, experiment_name, source, labeled_samples_source, target, labeled_samples_target):
		self.experiment_name = experiment_name
		self.source_name = source
		self.target_name = target
		self.labeled_source = labeled_samples_source
		self.labeled_target = labeled_samples_target
		self.params = self.get_parameters()
		self.source_loader, self.target_loader = self.get_loaders(train=True)
		self.source_loader_test, self.target_loader_test = self.get_loaders(train=False)


	def get_parameters(self):
		with open('./{}/config.json'.format(self.experiment_name), 'r') as f:
			params = json.load(f)
		return params

	def get_loaders(self, train):

		if self.experiment_name == 'digits':
			source_loader, target_loader = self.get_digits(train)

		return source_loader, target_loader


	def get_digits(self, train):

		self.transform = transforms.Compose([
		                 transforms.Resize(32),
		                 transforms.ToTensor(),
		                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		if self.source_name == 'mnist' and self.target_name == 'svhn':
			source_loader = self.get_mnist(train)
			target_loader = self.get_svhn(train)

		return source_loader, target_loader


	def get_mnist(self, train):

		mnist = datasets.MNIST(root='./digits/mnist/', download=False, transform=self.transform, train=train)
		loader = torch.utils.data.DataLoader(dataset=mnist,
		                                     batch_size=self.params['batch_size'],
		                                     shuffle=True,
		                                     num_workers=self.params['num_workers'])
		return loader


	def get_svhn(self, train):

		if train:
			split = 'train'
		else:
			split = 'test'

		svhn = datasets.SVHN(root='./digits/svhn/', download=False, transform=self.transform, split=split)
		loader = torch.utils.data.DataLoader(dataset=svhn,
		                                     batch_size=self.params['batch_size'],
		                                     shuffle=True,
		                                     num_workers=self.params['num_workers'])

		return loader

