import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim=512, ef_dim=64):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.ef_dim = ef_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, ef_dim, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(ef_dim)

        self.conv2 = nn.Conv2d(ef_dim, ef_dim * 2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(ef_dim * 2)

        self.conv3 = nn.Conv2d(ef_dim * 2, ef_dim * 4, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(ef_dim * 4)

        self.conv4 = nn.Conv2d(ef_dim * 4, ef_dim * 8, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(ef_dim * 8)

        # Flatten and fully connected layers for mean and log variance
        self.fc_mean = nn.Linear(ef_dim * 8 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(ef_dim * 8 * 4 * 4, z_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten

        # Latent mean and log variance
        z_mean = self.fc_mean(x)
        z_logvar = F.softplus(self.fc_logvar(x)) + 1e-6
        return z_mean, z_logvar

class Decoder(nn.Module):
    def __init__(self, z_dim=512, gf_dim=64, output_size=64, c_dim=3):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        s8 = output_size // 8  # 8, assuming the final upsampled size

        # Fully connected layer and reshape
        self.fc = nn.Linear(z_dim, gf_dim * 8 * s8 * s8)

        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(gf_dim * 4)

        self.deconv2 = nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(gf_dim * 2)

        self.deconv3 = nn.ConvTranspose2d(gf_dim * 2, gf_dim // 2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(gf_dim // 2)

        self.deconv4 = nn.ConvTranspose2d(gf_dim // 2, c_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, z):
        x = F.relu(self.fc(z)).view(-1, self.gf_dim * 8, 8, 8)  # Reshape

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))

        x = torch.tanh(self.deconv4(x))  # Output layer with tanh activation
        return x

class VAE(nn.Module):
    def __init__(self, z_dim=512, ef_dim=64, gf_dim=64, output_size=64, c_dim=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, ef_dim)
        self.decoder = Decoder(z_dim, gf_dim, output_size, c_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, logvar