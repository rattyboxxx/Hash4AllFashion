import torch
import torch.nn as nn

import utils.config as cfg


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if m.weight is not None:
            m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()


class LatentCode(nn.Module):
    """Basic class for learning latent code."""

    def __init__(self, param):
        """Latent code.
        
        Parameters:
        -----------
        See utils.param.NetParam
        """
        super().__init__()
        self.param = param
        self.register_buffer("scale", torch.ones(1))

    def set_scale(self, value):
        """Set the scale of tanh layer."""
        self.scale.fill_(value)

    def feat(self, x):
        """Compute the feature of all images."""
        raise NotImplementedError

    def forward(self, x):
    
        """Forward a feature from DeepContent."""
        x = self.feat(x)
        if self.param.without_binary:
            return x
        if self.param.scale_tanh:
            x = torch.mul(x, self.scale)
        if self.param.binary01:
            return 0.5 * (torch.tanh(x) + 1)
        # shape N x D
        return torch.tanh(x).view(-1, self.param.dim)


class ImgEncoder(LatentCode):
    """Module for encode to learn the latent code."""

    def __init__(self, in_feature, param):
        """Initialize an encoder.

        Parameter
        ---------
        in_feature: feature dimension for image features
        param: see utils.param.NetParam for details

        """
        super().__init__(param)
        half = in_feature // 2
        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_feature, half),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(half, param.dim, bias=False)
        )

    def feat(self, x):
        return self.encoder(x)

    def init_weights(self):
        """Initialize weights for encoder with pre-trained model."""
        nn.init.normal_(self.encoder[1].weight.data, std=0.01)
        nn.init.constant_(self.encoder[1].bias.data, 0)
        nn.init.normal_(self.encoder[-1].weight.data, std=0.01)


class CoreMat(nn.Module):
    """Weighted hamming similarity."""

    def __init__(self, dim):
        """Weights for this layer that is drawn from N(mu, std)."""
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.weight.data.fill_(1.0)

    def forward(self, x):
        """Forward."""
        return torch.mul(x, self.weight)

    def __repr__(self):
        """Format string for module CoreMat."""
        return self.__class__.__name__ + "(dim=" + str(self.dim) + ")"


class LearnableScale(nn.Module):
    def __init__(self, init=1.0):
        super(LearnableScale, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1).fill_(init))

    def forward(self, inputs):
        return self.weight * inputs

    def init_weights(self):
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"