import torch
import numpy as np
from models import *
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

plt.switch_backend('agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SSGAN:

	def __init__(self, data):
		self.data = data
		self.params = self.data.params
		self.source_loader = self.data.source_loader_sup
		self.target_loader = self.data.target_loader_sup
		self.source_loader_test = self.data.source_loader_test
		self.target_loader_test = self.data.target_loader_test
		self.criterion = nn.CrossEntropyLoss()
		self.build_model()


	def build_model(self):
		self.Gxy = Gxy().to(device)
		self.Gyx = Gyx().to(device)
		self.Dx = Dx().to(device)
		self.Dy = Dy().to(device)

		G_params = list(self.Gxy.parameters()) + list(self.Gyx.parameters())
		D_params = list(self.Dx.parameters()) + list(self.Dy.parameters())

		self.G_optimizer = optim.Adam(G_params, self.params['lr'], [self.params['beta1'], self.params['beta2']])
		self.D_optimizer = optim.Adam(D_params, self.params['lr'], [self.params['beta1'], self.params['beta2']])


	def reset_grad(self):
		self.G_optimizer.zero_grad()
		self.D_optimizer.zero_grad()


	def to_device(self):
		self.img_x, self.label_x = self.img_x.to(device), self.label_x.to(device)
		self.img_y, self.label_y = self.img_y.to(device), self.label_y.to(device)


	def train(self):
		nb_batches = min(len(self.source_loader), len(self.target_loader))

		for epoch in range(self.params['num_epochs']):
			print ('Epoch: {}'.format(epoch+1))
			source_iterator = iter(self.source_loader)
			target_iterator = iter(self.target_loader)
			for batch in range(nb_batches):
				self.img_x, self.label_x = source_iterator.next()
				self.img_y, self.label_y = target_iterator.next()
				self.to_device()

				self.train_discriminators()
				self.train_generators()

			self.save_checkpoint()
			self.plot_images(6, epoch)


	def train_discriminators(self):
		# Real loss
		self.reset_grad()
		out = self.Dx(self.img_x)
		d_x_loss = self.criterion(out, self.label_x)

		out = self.Dy(self.img_y)
		d_y_loss = self.criterion(out, self.label_y.long())

		real_loss = d_x_loss + d_y_loss
		real_loss.backward()
		self.D_optimizer.step()

		#Fake loss
		self.reset_grad()
		fake_label_x = Variable(torch.ones(self.img_x.size(0))*self.params['num_classes']).long().to(device)
		fake_img_x = self.Gyx(self.img_y)
		out = self.Dx(fake_img_x)
		d_x_loss = self.criterion(out, fake_label_x)

		fake_label_y = Variable(torch.ones(self.img_y.size(0))*self.params['num_classes']).long().to(device)
		fake_img_y = self.Gxy(self.img_x)
		out = self.Dy(fake_img_y)
		d_y_loss = self.criterion(out, fake_label_y)

		fake_loss = d_x_loss + d_y_loss
		fake_loss.backward()
		self.D_optimizer.step()


	def train_generators(self):
		# X -> Y -> X cycle
		self.reset_grad()

		fake_img_y = self.Gxy(self.img_x)
		out = self.Dy(fake_img_y)
		x_y_x_loss = self.criterion(out, self.label_x)
		img_reconst = self.Gyx(fake_img_y)
		x_y_x_loss += torch.mean((self.img_x - img_reconst)**2)

		x_y_x_loss.backward()
		self.G_optimizer.step()

		# Y -> X -> Y cycle
		self.reset_grad()

		fake_img_x = self.Gyx(self.img_y)
		out = self.Dx(fake_img_x)
		y_x_y_loss = self.criterion(out, self.label_y.long())
		img_reconst = self.Gxy(fake_img_x)
		y_x_y_loss += torch.mean((self.img_y - img_reconst)**2)

		y_x_y_loss.backward()
		self.G_optimizer.step()


	def plot_images(self, nb_images, epoch):
		images = torch.stack([self.source_loader_test.dataset[i][0] for i in range(nb_images)], dim = 0).to(device)
		labels = torch.stack([self.source_loader_test.dataset[i][1] for i in range(nb_images)], dim = 0).to(device)
		self.Gxy.eval()
		self.Dy.eval()
		fake_images = self.Gxy(images)
		fake_labels = np.argmax(self.Dy(fake_images).cpu().detach().numpy(), axis=1)
		
		f, ax = plt.subplots(2, 3)
		for i in range(nb_images):
			fake_im = (fake_images[i].cpu().detach().numpy().transpose(1,2,0) + 1)/2.
			ax[int(i/3),i%3].imshow(fake_im)
			ax[int(i/3),i%3].set_title('Fake: {}'.format(fake_labels[i]))
		plt.savefig('./samples/epoch_{}'.format(epoch))


	def save_checkpoint(self):
		state = {'state_dict': self.Gxy.state_dict()}
		torch.save(state, 'Gxy_model.pth.tar')

