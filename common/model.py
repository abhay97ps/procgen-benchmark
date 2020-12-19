from .misc_util import orthogonal_init, xavier_uniform_init
import torch.nn as nn
import torch
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class MlpModel(nn.Module):
    def __init__(self,
                 input_dims=4,
                 hidden_dims=[64, 64],
                 **kwargs):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(MlpModel, self).__init__()

        # Hidden layers
        hidden_dims = [input_dims] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self.apply(orthogonal_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NatureModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(NatureModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=64*7*7, out_features=512), nn.ReLU()
        )
        self.output_dim = 512
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16)
        self.block2 = ImpalaBlock(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlock(in_channels=32, out_channels=32)
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru = orthogonal_init(nn.GRU(input_size, hidden_size), gain=1.0)

    def forward(self, x, hxs, masks):
        # Prediction
        if x.size(0) == hxs.size(0):
            # input for GRU-CELL: (L=sequence_length, N, H)
            # output for GRU-CELL: (output: (L, N, H), hidden: (L, N, H))
            masks = masks.unsqueeze(-1)
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        # Training
        # We will recompute the hidden state to allow gradient to be back-propagated through time
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # (can be interpreted as a truncated back-propagation through time)
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
    
class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features):
        super(LinearAttentionBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) 
        # out N, 1, W, H
        a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        g = torch.mul(a.expand_as(l), l)
        g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        return c.view(N,1,W,H), g

class AttentionModel(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(AttentionModel, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, 8, 2)
        self.conv_block2 = ConvBlock(8, 16, 2)
        self.conv_block3 = ConvBlock(16, 32, 2)
        self.conv_block4 = ConvBlock(32, 64, 3)
        self.conv_block5 = ConvBlock(64, 64, 3)
        self.conv_block6 = ConvBlock(64, 64, 3, pool=True)
        self.dense = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, bias=True)

        # for l1 attetnion
        self.projector = nn.Conv2d(32, 64, kernel_size=1, padding=0, bias=False)
        self.att1 = LinearAttentionBlock(64)
        self.att2 = LinearAttentionBlock(64)
        self.att3 = LinearAttentionBlock(64)

        self.output_dim = 256
        self.embed = nn.Linear(in_features=64*4, out_features=self.output_dim)

    def forward(self, x):
        # input N, 3, 64, 64
        x = self.conv_block1(x)
        # out N, 8, 64, 64
        x = self.conv_block2(x)
        # out N, 16, 64, 64
        x = self.conv_block3(x)
        # out N, 32, 64, 64
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        # out N, 32, 32, 32
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2, padding=0)
        # out N, 64, 16, 16
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2, padding=0)
        # out N, 64, 8, 8
        x = self.conv_block6(l3)
        # out N, 64, 1, 1
        g = self.dense(x)
        # out N, 64, 1, 1
        c1, g1 = self.att1(self.projector(l1), g)
        c2, g2 = self.att2(l2, g)
        c3, g3 = self.att3(l3, g)
        N, C, _, _ = g.size()
        g = g.view(N,C,-1).sum(dim=2)
        g = torch.cat((g,g1,g2,g3), dim=1)
        g = self.embed(g)
        return [g,c1,c2,c3]