import torch
import torch.nn as nn



OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: AvgPool2d(stride),
  'max_pool_3x3' : lambda C, stride, affine: MaxPool2d(stride),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv3(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv5(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv7(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv3(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv5(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: Conv7(C,C,(1,7),stride,3,affine=affine),
}

 

class AvgPool2d(nn.Module):
  def __init__(self, stride):
    super(AvgPool2d, self).__init__()
    self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
    self.count = 0
    self.pre_grads = []
    self.avg = 0

  def forward(self, x):
    return self.op(x)

class MaxPool2d(nn.Module):
  def __init__(self, stride):
    super(MaxPool2d, self).__init__()
    self.op = nn.MaxPool2d(3, stride=stride, padding=1)
    self.count = 0
    self.pre_grads=[]
    self.avg = 0
    
  def forward(self, x):
    return self.op(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )
  def forward(self, x):
    return self.op(x)

class DilConv3(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv3, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.count = 0
    self.pre_grads=[]
    self.avg = 0

  def forward(self, x):
    return self.op(x)

class DilConv5(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv5, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.count = 0
    self.pre_grads=[]
    self.avg = 0

  def forward(self, x):
    return self.op(x)

class SepConv3(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv3, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.count = 0
    self.pre_grads=[]
    self.avg = 0

  def forward(self, x):
    return self.op(x)

class SepConv5(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv5, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.count = 0
    self.pre_grads=[]
    self.avg = 0

  def forward(self, x):
    return self.op(x)

class SepConv7(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv7, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )
    self.count = 0
    self.pre_grads=[]
    self.avg = 0

  def forward(self, x):
    return self.op(x)

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
    self.count = 0
    self.pre_grads=[]

  def forward(self, x):
    return x

class Zero(nn.Module):
  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride
    self.count = 0
    self.pre_grads=[]
    self.avg = 0

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.count = 0
    self.pre_grads=[]
    self.avg = 0

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

class Conv7(nn.Module):
  def __init__(self, C_in, C_out, stride, affine=True):
    super(Conv7, self).__init__()
    self.op = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(C_in, C_in, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
                nn.Conv2d(C_in, C_out, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
                )

    self.count = 0
    self.pre_grads=[]
    self.avg = 0
  
  def forward(self, x):
    return self.op(x)
