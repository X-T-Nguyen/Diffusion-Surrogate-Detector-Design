# ModelCondition.py
import math
import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.timembedding(t)

# Discrete conditional embedding (used for energy labels)
class ConditionalEmbedding(nn.Module):
    def __init__(self, num_labels, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        # num_embeddings is num_labels + 1 to allow a null condition if needed.
        self.condEmbedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        return self.condEmbedding(t)

# New continuous embedding module for xy and z labels.
class ContinuousEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            Swish(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, x):
        # Ensure x has shape (batch_size, 1)
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return self.fc(x)

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_ch, in_ch, 5, stride=2, padding=2)

    def forward(self, x, temb, cemb):
        return self.c1(x) + self.c2(x)

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.t = nn.ConvTranspose2d(in_ch, in_ch, 5, 2, 2, 1)

    def forward(self, x, temb, cemb):
        x = self.t(x)
        return self.c(x)

class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h).permute(0, 2, 3, 1).view(B, H * W, C)
        k = self.proj_k(h).view(B, C, H * W)
        w = torch.bmm(q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=-1)
        v = self.proj_v(h).permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v).view(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.proj(h)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.cond_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = AttnBlock(out_ch) if attn else nn.Identity()

    def forward(self, x, temb, labels):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None] + self.cond_proj(labels)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attn(h)

class UNet(nn.Module):
    def __init__(self, T, num_energy_labels, ch, ch_mult, num_res_blocks, dropout):
        """
        Note: num_energy_labels remains discrete, while xy, z and material will be treated as continuous.
        """
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.energy_embedding = ConditionalEmbedding(num_energy_labels, ch, tdim)
        self.xy_embedding = ContinuousEmbedding(1, tdim)
        self.z_embedding = ContinuousEmbedding(1, tdim)
        # New material embedding:
        self.material_embedding = ContinuousEmbedding(1, tdim)
        
        self.head = nn.Conv2d(1, ch, 3, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(now_ch, out_ch, tdim, dropout))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
                
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])
        
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(chs.pop() + now_ch, out_ch, tdim, dropout, attn=False))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0
        
        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 1, 3, padding=1)
        )

    def forward(self, x, t, energy_labels, xy_labels, z_labels, material_labels):
        temb = self.time_embedding(t)
        e_emb = self.energy_embedding(energy_labels)
        xy_emb = self.xy_embedding(xy_labels)
        z_emb = self.z_embedding(z_labels)
        # Compute material embedding
        m_emb = self.material_embedding(material_labels)
        # Combine all conditioning embeddings:
        cemb = e_emb + xy_emb + z_emb + m_emb

        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb)
            hs.append(h)
        for layer in self.middleblocks:
            h = layer(h, temb, cemb)
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb, cemb)
        h = self.tail(h)
        assert len(hs) == 0
        return h

if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000,
        num_energy_labels=11,    # for discrete energy labels
        ch=32,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        dropout=0.1
    )
    x = torch.randn(batch_size, 1, 32, 32)
    t = torch.randint(1000, size=[batch_size])
    # For testing, generate continuous random values for xy and z:
    energy_labels = torch.randint(5, size=[batch_size])
    xy_labels = torch.rand(batch_size)
    z_labels = torch.rand(batch_size)
    y = model(x, t, energy_labels, xy_labels, z_labels)
    print(y.shape)
