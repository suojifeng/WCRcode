import torch
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

# Pixel-set encoder for optical data
class OpticalEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OpticalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 4 * 4, out_channels)

    def forward(self, x):
        # x shape: (batch_size, in_channels, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x shape: (batch_size, 64, 8, 8)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x shape: (batch_size, 128, 4, 4)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # x shape: (batch_size, 256, 2, 2)
        x = x.view(x.size(0), -1)
        # x shape: (batch_size, 256 * 4 * 4)
        x = self.fc(x)
        # x shape: (batch_size, out_channels)
        return x

# Pixel-set encoder for SAR data
class SAREncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAREncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(256 * 4 * 4, out_channels)

    def forward(self, x):
        # x shape: (batch_size, in_channels, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # x shape: (batch_size, 64, 8, 8)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # x shape: (batch_size, 128, 4, 4)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # x shape: (batch_size, 256, 2, 2)
        x = x.view(x.size(0), -1)
        # x shape: (batch_size, 256 * 4 * 4)
        x = self.fc(x)
        # x shape: (batch_size, out_channels)
        return x

# Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(out_channels, num_heads), num_layers)

    def forward(self, x1, x2):
        # x1 shape: (batch_size, in_channels)
        # x2 shape: (batch_size, in_channels)
        x = torch.cat((x1, x2), dim=1)
        # x shape: (batch_size, in_channels * 2)
        x = self.linear(x)
        # x shape: (batch_size, out_channels)
        x = x.unsqueeze(0)
        # x shape: (1, batch_size, out_channels)
        x = self.transformer(x)
        # x shape: (1, batch_size, out_channels)
        x = x.squeeze(0)
        # x shape: (batch_size, out_channels)
        return x

# Projection head
class ProjectionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.fc2 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        # x shape: (batch_size, in_channels)
        x = F.relu(self.fc1(x))
        # x shape: (batch_size, in_channels)
        x = self.fc2(x)
        # x shape: (batch_size, out_channels)
        return x

# Pixel-Set encoder + Temporal Attention Encoder sequence classifier
class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=10, mlp1=[10, 32, 64], pooling='mean_std', mlp2=[132, 128], with_extra=True,
                 extra_size=4,
                 n_head=4, d_k=32, d_model=None, mlp3=[512, 128, 128], dropout=0.2, T=1000, len_max_seq=24,
                 positions=None,
                 mlp4=[128, 64, 32, 20]):
        super(PseTae, self).__init__()
        # Modify the output dimension of pixel-set encoder to match the input dimension of transformer encoder
        mlp2[-1] = d_model
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp1=mlp1, pooling=pooling, mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        # Modify the input dimension of temporal attention encoder to match the output dimension of projection head
        in_channels = mlp4[0]
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp4[0], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=len_max_seq, positions=positions)
        self.decoder = get_decoder(mlp4)
        self.name = '_'.join([self.spatial_encoder.name, self.temporal_encoder.name])

    def forward(self, input):
        """
         Args:
            input(tuple): (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features)
            Pixel-Set : Batch_size x Sequence length x Channel x Number of pixels
            Pixel-Mask : Batch_size x Sequence length x Number of pixels
            Extra-features : Batch_size x Sequence length x Number of features
        """
        # Define the pixel-set encoder for optical data
        optical_encoder = OpticalEncoder(input_dim, mlp2[-1])
        # Define the pixel-set encoder for SAR data
        sar_encoder = SAREncoder(input_dim, mlp2[-1])
        # Define the transformer encoder
        transformer_encoder = TransformerEncoder(mlp2[-1], d_model, n_head, num_layers)
        # Define the projection head
        projection_head = ProjectionHead(d_model, mlp4[0])