import os
import torch
import json
import numpy as np
import gzip, pickle
from PIL import Image
from scipy.io import loadmat
from torch.utils import data
from torchvision import datasets
from torchvision import transforms



class Preprocess:

	def __init__(self, exp_name, source, labeled_samples_source, target, labeled_samples_target, balanced_s=False, balanced_t=True):
		self.exp_name = exp_name
		self.source_name = source
		self.target_name = target
		self.labeled_source = labeled_samples_source
		self.labeled_target = labeled_samples_target
		self.balanced_s = balanced_s
		self.balanced_t = balanced_t
		self.params = self.get_parameters()
		self.source_loader_unsup, self.target_loader_unsup = self.get_loaders()
		self.source_loader_sup, self.target_loader_sup = self.get_loaders(sup=True)
		self.source_loader_test, self.target_loader_test = self.get_loaders(train=False)


	def get_parameters(self):

		with open('./{}/config.json'.format(self.exp_name), 'r') as f:
			params = json.load(f)

		return params


	def get_loaders(self, train=True, sup=False):

		if self.exp_name == 'digits':
			source_loader, target_loader = self.get_digits(train, sup)

		return source_loader, target_loader


	def get_digits(self, train, sup):

		self.transform = transforms.Compose([
		                 transforms.Resize(32),
		                 transforms.ToTensor(),
		                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		if self.source_name == 'mnist' and self.target_name == 'svhn':
			source_loader = self.get_mnist(train, sup)
			target_loader = self.get_svhn(train, sup, 'target')

		elif self.source_name == 'usps' and self.target_name == 'svhn':
			source_loader = self.get_usps(train, sup)
			target_loader = self.get_svhn(train, sup, 'target')

		return source_loader, target_loader


	def get_mnist(self, train, sup, domain='source'):

		batch_size = self.params['batch_size']

		if train:
			X, y = torch.load(os.path.join('./digits/mnist/processed/', 'training.pt'))
		else:
			X, y = torch.load(os.path.join('./digits/mnist/processed/', 'test.pt'))
			batch_size = X.size(0)

		if sup:
			X, y = self.get_labeled_samples(X, y.numpy(), domain)

		mnist = CustomDataset(X.numpy(), y, self.transform)

		loader = torch.utils.data.DataLoader(dataset=mnist,
		                                     batch_size=batch_size,
		                                     shuffle=True,
		                                     num_workers=self.params['num_workers'],
		                                     worker_init_fn=0)
		return loader


	def get_usps(self, train, sup, domain='source', repeat=False):

		batch_size = self.params['batch_size']

		if train:
			with gzip.open('digits/usps/usps_28x28.pkl', 'rb') as f:
				(X, y), (_, _) = pickle.load(f, encoding='bytes')
		else:
			with gzip.open('digits/usps/usps_28x28.pkl', 'rb') as f:
				(_, _), (X, y) = pickle.load(f, encoding='bytes')
				batch_size = X.shape[0]

		X *= 255.0

		if sup:
			X, y = self.get_labeled_samples(X, y, domain)
	
		X = X.reshape(X.shape[0], 28, 28).astype('uint8')

		usps = CustomDataset(X, y, self.transform)

		loader = torch.utils.data.DataLoader(dataset=usps,
		                                     batch_size=batch_size,
		                                     shuffle=True,
		                                     num_workers=self.params['num_workers'],
		                                     worker_init_fn=0)
		return loader


	def get_svhn(self, train, sup, domain='source', repeat=False):

		batch_size = self.params['batch_size']

		if train:
			data = loadmat('digits/svhn/train_32x32.mat')
			X, y = data['X'].transpose(-1,0,1,2), data['y']
			y[y == 10] = 0
		else:
			data = loadmat('digits/svhn/test_32x32.mat')
			X, y = data['X'].transpose(-1,0,1,2), data['y']
			y[y == 10] = 0
			batch_size = X.shape[0]

		y = y.reshape(y.shape[0],)
		svhn = CustomDataset(X, y, self.transform)

		loader = torch.utils.data.DataLoader(dataset=svhn,
		                                     batch_size=batch_size,
		                                     shuffle=True,
		                                     num_workers=self.params['num_workers'],
		                                     worker_init_fn=0)
		return loader


	def get_labeled_samples(self, X, y, domain):

		if domain == 'source':
			if self.balanced_s:
				X, y = self.get_labeled_samples_balanced(X, y, self.labeled_source)
			else:
				X, y = self.get_labeled_samples_unbalanced(X, y, self.labeled_source)

		elif domain == 'target':
			if self.balanced_t:
				X, y = self.get_labeled_samples_balanced(X, y, self.labeled_target)
			else:
				X, y = self.get_labeled_samples_unbalanced(X, y, self.labeled_target)

		return X, y


	def get_labeled_samples_balanced(self, X, y, samples_to_label):

		np.random.seed(0)

		classes = np.unique(y)
		indxs = [np.where(y == class_) for class_ in classes]

		ix = []
		labels_per_class = int(samples_to_label/len(classes))
		for indx in indxs:
			ix.extend(np.random.choice(indx[0], labels_per_class, replace = False))

		np.random.shuffle(ix)
		X_sup = X[ix]
		y_sup = y[ix]

		return X_sup, y_sup


	def get_labeled_samples_unbalanced(self, X, y, samples_to_label):

		np.random.seed(0)
		ix = np.random.choice(len(X), samples_to_label, replace = False)
		X_sup = X[ix]
		y_sup = y[ix]

		return X_sup, y_sup



class CustomDataset(data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)

        return x, y
    
    def __len__(self):
        return len(self.data)

