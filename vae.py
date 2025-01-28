import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels=1):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(
            # first conv
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # second conv
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # third conv
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # fourth conv
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.hidden_dim = 128 * 16 * 16

        # latent space
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_gvar = nn.Linear(self.hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.hidden_dim)

        self.decoder = nn.Sequential(
            # first deconv
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            # second deconv
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            # third deconv
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            # fourth deconv
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.ReLU(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        gvar = self.fc_gvar(x)
        return mu, gvar

    def reparametrize(self, mu, gvar):
        std = torch.exp(0.5 * gvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 128, 16, 16)
        return self.decoder(z)

    def forward(self, x):
        mu, gvar = self.encode(x)
        z = self.reparametrize(mu, gvar)
        return self.decode(z), mu, gvar


def loss_function(recon_x, x, mu, log_var):
    # Reconstruction loss
    MSE = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return MSE + KLD
