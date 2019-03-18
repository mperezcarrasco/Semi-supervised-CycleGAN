from preprocess import Preprocess
from train import SSGAN




if __name__ == '__main__':
	data = Preprocess('digits', 'mnist', 20000, 'svhn', 20000, balanced_t=False)
	ssgan = SSGAN(data)
	ssgan.train()
