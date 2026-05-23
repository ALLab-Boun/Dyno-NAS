import torch
import torch.nn as nn

OPS = {
  'none': lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'mbconv': lambda C, stride, affine: MBConv(C, C, stride, expansion_ratio=6, affine=affine),
}


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


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, groups=C_in, bias=False),
          nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
          nn.BatchNorm2d(C_out, affine=affine),
          )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
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

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

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

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    def __init__(self, channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.squeeze(x).view(batch, channels)
        # Excitation: FC layers with ReLU and Sigmoid
        y = self.excitation(y).view(batch, channels, 1, 1)
        # Scale: Reweight the feature maps
        return x * y.expand_as(x)


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution with Squeeze-and-Excitation
    Reference: 
    - Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018
    - Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    
    This combines the efficient inverted residual structure from MobileNetV2
    with the channel attention mechanism from SE-Net.
    """
    def __init__(self, C_in, C_out, stride, expansion_ratio=6, affine=True, use_se=True):
        super(MBConv, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and C_in == C_out)
        
        hidden_dim = C_in * expansion_ratio
        
        layers = []
        
        # Expansion phase (Pointwise)
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(C_in, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, affine=affine),
                nn.ReLU6(inplace=False)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, affine=affine),
            nn.ReLU6(inplace=False)
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Squeeze-and-Excitation
        self.se = SEBlock(hidden_dim, reduction_ratio=4) if use_se else None
        
        # Projection phase (Pointwise linear)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, C_out, 1, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        identity = x
        
        out = self.conv(x)
        
        if self.se is not None:
            out = self.se(out)
        
        out = self.project(out)
        
        if self.use_residual:
            out = out + identity
            
        return out
