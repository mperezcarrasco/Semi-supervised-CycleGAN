import torch.nn as nn
import torch.nn.functional as F



class Gxy(nn.Module):
    def __init__(self, conv_dim=64):
        super(Gxy, self).__init__()
        self.cnn1 = Conv(3, conv_dim, 4, 2, 1)
        self.cnn2 = Conv(conv_dim, conv_dim*2, 3, 2, 1)
        self.cnn3 = Conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.cnn4 = Conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.cnn5 = Deconv(conv_dim*2, conv_dim, 4, 2, 1)
        self.cnn6 = Deconv(conv_dim, 3, 4, 2, 1, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.cnn1(x), 0.05) 
        out = F.leaky_relu(self.cnn2(out), 0.05)
        
        out = F.leaky_relu(self.cnn3(out), 0.05)
        out = F.leaky_relu(self.cnn4(out), 0.05)
        
        out = F.leaky_relu(self.cnn5(out), 0.05)
        out = F.tanh(self.cnn6(out))
        return out

class Gyx(nn.Module):
    def __init__(self, conv_dim=64):
        super(Gyx, self).__init__()
        self.cnn1 = Conv(3, conv_dim, 4, 2, 1)
        self.cnn2 = Conv(conv_dim, conv_dim*2, 3, 2, 1)
        self.cnn3 = Conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.cnn4 = Conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.cnn5 = Deconv(conv_dim*2, conv_dim, 4, 2, 1)
        self.cnn6 = Deconv(conv_dim, 3, 4, 2, 1, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.cnn1(x), 0.05) 
        out = F.leaky_relu(self.cnn2(out), 0.05)
        
        out = F.leaky_relu(self.cnn3(out), 0.05)
        out = F.leaky_relu(self.cnn4(out), 0.05)
        
        out = F.leaky_relu(self.cnn5(out), 0.05)
        out = F.tanh(self.cnn6(out))
        return out
    
class Dx(nn.Module):
    def __init__(self, conv_dim=64):
        super(Dx, self).__init__()
        self.cnn1 = Conv(1, conv_dim, 4, 2, 1, bn=False)
        self.cnn2 = Conv(conv_dim, conv_dim*2, 4, 2, 1)
        self.cnn3 = Conv(conv_dim*2, conv_dim*4, 4, 2, 1)
        self.fc = Conv(conv_dim*4, 11, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.cnn1(x), 0.05)   
        out = F.leaky_relu(self.cnn2(out), 0.05)
        out = F.leaky_relu(self.cnn3(out), 0.05)
        out = self.fc(out).squeeze()
        return out

class Dy(nn.Module):
    def __init__(self, conv_dim=64):
        super(Dy, self).__init__()
        self.cnn1 = Conv(3, conv_dim, 4, 2, 1, bn=False)
        self.cnn2 = Conv(conv_dim, conv_dim*2, 4, 2, 1)
        self.cnn3 = Conv(conv_dim*2, conv_dim*4, 4, 2, 1)
        self.fc = Conv(conv_dim*4, 11, 4, 1, 0, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.cnn1(x), 0.05)
        out = F.leaky_relu(self.cnn2(out), 0.05)
        out = F.leaky_relu(self.cnn3(out), 0.05)
        out = self.fc(out).squeeze()
        return out


class Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bn=True):
        super(Conv, self).__init__()
        self.bn = bn
        self.conv2d = nn.Conv2d(in_channels=dim_in, out_channels= dim_out,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=True)
        self.bn2d = nn.BatchNorm2d(num_features=dim_out)
    def forward(self, x):
        if self.bn:
            return self.bn2d(self.conv2d(x))
        else:
            return self.conv2d(x)


class Deconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bn=True):
        super(Deconv, self).__init__()
        self.bn = bn
        self.deconv2d = nn.ConvTranspose2d(in_channels=dim_in, out_channels=dim_out, 
                                           kernel_size=kernel_size, stride=stride, 
                                           padding=padding, bias=True) 
        self.bn2d = nn.BatchNorm2d(num_features=dim_out)
    def forward(self, x):
        if self.bn:
            return self.bn2d(self.deconv2d(x))
        else: 
            return self.deconv2d(x)
