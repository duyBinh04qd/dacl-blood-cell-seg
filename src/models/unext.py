"""UNeXt: MLP-based Rapid Medical Image Segmentation Network (Valanarasu & Patel, MICCAI 2022).

Re-implemented from the official repo (jeya-maria-jose/UNeXt-pytorch) with configurable channels.
DACL states encoder channels {32, 64, 128, 160, 256} (see UNEXT_DIMS in __init__.py).

Architecture:
  - 3 convolutional stages (Conv -> BN -> MaxPool -> ReLU), each halving the resolution
  - 2 "Tokenized MLP" (shifted MLP) stages at depth + bottleneck
  - Symmetric decoder; skip connections use ADDITION instead of concatenation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_


class DWConv(nn.Module):
    """3x3 depth-wise convolution over the token sequence (implicit positional encoding)."""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class ShiftMLP(nn.Module):
    """Shifted MLP: shift channel groups along height, then width, before each FC layer,
    so the MLP "sees" a local neighbourhood like a convolution, but cheaper."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2

    def _shift(self, x, H, W, dim):
        """Split channels into shift_size groups; group k is rolled by k-pad pixels along axis dim."""
        B, N, C = x.shape
        xn = x.transpose(1, 2).reshape(B, C, H, W)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        chunks = torch.chunk(xn, self.shift_size, 1)
        shifted = [torch.roll(c, s, dim) for c, s in zip(chunks, range(-self.pad, self.pad + 1))]
        xs = torch.cat(shifted, 1)
        xs = torch.narrow(xs, 2, self.pad, H)
        xs = torch.narrow(xs, 3, self.pad, W)
        return xs.reshape(B, C, H * W).transpose(1, 2)

    def forward(self, x, H, W):
        x = self._shift(x, H, W, dim=2)  # shift along height
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self._shift(x, H, W, dim=3)  # shift along width
        x = self.fc2(x)
        return self.drop(x)


class ShiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=1.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ShiftMLP(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, H, W):
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class OverlapPatchEmbed(nn.Module):
    """3x3 stride-2 conv -> halves the resolution and turns the feature map into tokens."""

    def __init__(self, in_chans, embed_dim, patch_size=3, stride=2):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H, W


class UNext(nn.Module):
    def __init__(self, num_classes=2, in_chans=3, dims=(32, 64, 128, 160, 256), drop_path_rate=0.0):
        """
        num_classes: number of output classes (2 = background + cell).
        dims: channels of the 5 encoder stages. The first 3 are the conv stages,
              the last 2 the tokenized-MLP stage and the bottleneck.
              Original UNeXt: (16, 32, 128, 160, 256); DACL: (32, 64, 128, 160, 256).
        """
        super().__init__()
        c1, c2, c3, c4, c5 = dims
        # ---- Encoder: convolutional stages ----
        self.encoder1 = nn.Conv2d(in_chans, c1, 3, 1, 1)
        self.encoder2 = nn.Conv2d(c1, c2, 3, 1, 1)
        self.encoder3 = nn.Conv2d(c2, c3, 3, 1, 1)
        self.ebn1, self.ebn2, self.ebn3 = nn.BatchNorm2d(c1), nn.BatchNorm2d(c2), nn.BatchNorm2d(c3)
        # ---- Encoder: tokenized MLP ----
        self.patch_embed3 = OverlapPatchEmbed(c3, c4)
        self.patch_embed4 = OverlapPatchEmbed(c4, c5)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 3)]
        self.block1 = nn.ModuleList([ShiftedBlock(c4, drop_path=dpr[0])])
        self.block2 = nn.ModuleList([ShiftedBlock(c5, drop_path=dpr[1])])
        self.norm3, self.norm4 = nn.LayerNorm(c4), nn.LayerNorm(c5)
        # ---- Decoder ----
        self.dblock1 = nn.ModuleList([ShiftedBlock(c4, drop_path=dpr[0])])
        self.dblock2 = nn.ModuleList([ShiftedBlock(c3, drop_path=dpr[1])])
        self.dnorm3, self.dnorm4 = nn.LayerNorm(c4), nn.LayerNorm(c3)
        self.decoder1 = nn.Conv2d(c5, c4, 3, 1, 1)
        self.decoder2 = nn.Conv2d(c4, c3, 3, 1, 1)
        self.decoder3 = nn.Conv2d(c3, c2, 3, 1, 1)
        self.decoder4 = nn.Conv2d(c2, c1, 3, 1, 1)
        self.decoder5 = nn.Conv2d(c1, c1, 3, 1, 1)
        self.dbn1, self.dbn2 = nn.BatchNorm2d(c4), nn.BatchNorm2d(c3)
        self.dbn3, self.dbn4 = nn.BatchNorm2d(c2), nn.BatchNorm2d(c1)
        self.final = nn.Conv2d(c1, num_classes, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _to_map(x, B, H, W):
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        B = x.shape[0]
        # Encoder - conv stages (1/2, 1/4, 1/8)
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2)); t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2)); t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2)); t3 = out
        # Tokenized MLP (1/16)
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self._to_map(self.norm3(out), B, H, W); t4 = out
        # Bottleneck (1/32)
        out, H, W = self.patch_embed4(out)
        for blk in self.block2:
            out = blk(out, H, W)
        out = self._to_map(self.norm4(out), B, H, W)

        # Decoder
        out = F.relu(self._up(self.dbn1(self.decoder1(out)))) + t4
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock1:
            out = blk(out, H, W)
        out = self._to_map(self.dnorm3(out), B, H, W)

        out = F.relu(self._up(self.dbn2(self.decoder2(out)))) + t3
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for blk in self.dblock2:
            out = blk(out, H, W)
        out = self._to_map(self.dnorm4(out), B, H, W)

        out = F.relu(self._up(self.dbn3(self.decoder3(out)))) + t2
        out = F.relu(self._up(self.dbn4(self.decoder4(out)))) + t1
        out = F.relu(self._up(self.decoder5(out)))
        return self.final(out)  # logits (B, num_classes, H, W)
