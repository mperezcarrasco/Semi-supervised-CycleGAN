from preprocess import Preprocess
from train import SSGAN




if __name__ == '__main__':
	data = Preprocess('digits', 'mnist', 100, 'svhn', 100)
	ssgan = SSGAN(data)

	ssgan.train()
