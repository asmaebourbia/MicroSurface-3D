"""Independent 64x64 conditional GAN; NOT a DG-MBSR derivative."""
import math
import torch
from torch import nn
from torch.nn.utils import spectral_norm

Z_DIM = 96
HEIGHT_SCALE = 2.5


def normalize_cond(c):
    # Conditions: [RMS slope, std(dh/dx) / std(dh/dy)].
    return torch.stack(((c[:, 0] - 0.18) / 0.12,
                        torch.log(c[:, 1].clamp(min=1e-6)) / math.log(3)), dim=1)


class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim + 2, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
    def forward(self, z, cond):
        x = torch.cat((z, normalize_cond(cond)), dim=1)
        h = self.net(self.fc(x).view(-1, 256, 4, 4)) * HEIGHT_SCALE
        return h - h.mean(dim=(2, 3), keepdim=True)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(32, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2),
        )
        self.score = spectral_norm(nn.Linear(256 * 4 * 4, 1))
        self.condition_embed = nn.Linear(2, 256 * 4 * 4)
    def forward(self, h, cond):
        f = self.features(h / HEIGHT_SCALE).flatten(1)
        projection = (f * self.condition_embed(normalize_cond(cond))).sum(1) / math.sqrt(f.size(1))
        return self.score(f).squeeze(1) + projection


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        # spectral_norm conv's trainable tensor is weight_orig.
        w = getattr(m, 'weight_orig', m.weight)
        nn.init.normal_(w, 0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def load_generator(checkpoint, device='cpu'):
    # Only load checkpoints created by this project / trusted sources.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    g = Generator(z_dim=state.get('z_dim', Z_DIM)).to(device)
    g.load_state_dict(state['generator'])
    g.eval()
    return g
