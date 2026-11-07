"""Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation (Cao et al., ECCV-W 2022).

Cài đặt lại theo repo gốc (HuCaoFighting/Swin-Unet), giữ nguyên tên tham số để nạp được
trọng số Swin-T tiền huấn luyện ImageNet của Microsoft.

Cấu hình "tiny" dùng trong DACL:
  embed_dim=96, depths=[2,2,2,2], num_heads=[3,6,12,24], window_size=7, ảnh 224x224
  -> đặc trưng [96, 192, 384] ở bộ mã hóa, 768 ở nút cổ chai, ~27M tham số.
"""
import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_

# Trọng số Swin-T (ImageNet-1K) chính thức của Microsoft - đúng file mà repo Swin-Unet sử dụng.
SWIN_TINY_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/"
    "swin_tiny_patch4_window7_224.pth"
)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Self-attention trong cửa sổ cục bộ, có bias vị trí tương đối."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size[0]), torch.arange(window_size[1]), indexing="ij"))
        coords = torch.flatten(coords, 1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(input_resolution) <= window_size:
            # Feature map nhỏ hơn cửa sổ -> không chia cửa sổ, không dịch.
            self.shift_size = 0
            self.window_size = min(input_resolution)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, to_2tuple(self.window_size), num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            cnt = 0
            for h in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                      slice(-self.shift_size, None)):
                for w in (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                          slice(-self.shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mw = window_partition(img_mask, self.window_size).view(-1, self.window_size ** 2)
            attn_mask = mw.unsqueeze(1) - mw.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        xw = window_partition(x, self.window_size).view(-1, self.window_size ** 2, C)
        aw = self.attn(xw, mask=self.attn_mask).view(-1, self.window_size, self.window_size, C)
        x = window_reverse(aw, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + self.drop_path(x.view(B, H * W, C))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """Giảm kích thước 2 lần, tăng kênh 2 lần (bộ mã hóa)."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C)))


class PatchExpand(nn.Module):
    """Tăng kích thước 2 lần, giảm kênh 2 lần (bộ giải mã)."""

    def __init__(self, input_resolution, dim, dim_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, 2, 2, C // 4).permute(0, 1, 3, 2, 4, 5).reshape(B, H * 2 * W * 2, C // 4)
        return self.norm(x)


class FinalPatchExpandX4(nn.Module):
    """Tăng kích thước 4 lần để trở về độ phân giải ảnh gốc."""

    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        c = C // 16
        x = x.view(B, H, W, 4, 4, c).permute(0, 1, 3, 2, 4, 5).reshape(B, H * 4 * W * 4, c)
        return self.norm(x)


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio, qkv_bias,
                 drop, attn_drop, drop_path, downsample=None, upsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim, input_resolution, num_heads, window_size,
                                 0 if i % 2 == 0 else window_size // 2,  # khối chẵn: W-MSA, lẻ: SW-MSA
                                 mlp_ratio, qkv_bias, drop, attn_drop,
                                 drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)])
        self.downsample = downsample(input_resolution, dim) if downsample is not None else None
        self.upsample = upsample(input_resolution, dim, 2) if upsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class SwinUnet(nn.Module):
    def __init__(self, img_size=224, num_classes=2, in_chans=3, embed_dim=96,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), window_size=7,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1):
        """
        img_size: kích thước ảnh vào. img_size/4, /8, /16 phải chia hết cho window_size
                  -> với window 7 dùng 224 (hoặc 448).
        drop_path_rate: Stochastic Depth, chống quá khớp (0.1 theo cấu hình gốc của Swin-Unet).
        """
        super().__init__()
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(img_size, 4, in_chans, embed_dim)
        pr = self.patch_embed.patches_resolution
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        L = self.num_layers

        # ---- Bộ mã hóa ----
        self.layers = nn.ModuleList()
        for i in range(L):
            self.layers.append(BasicLayer(
                int(embed_dim * 2 ** i), (pr[0] // 2 ** i, pr[1] // 2 ** i), depths[i], num_heads[i],
                window_size, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate,
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging if i < L - 1 else None))

        # ---- Bộ giải mã (đối xứng) ----
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(L):
            j = L - 1 - i
            dim = int(embed_dim * 2 ** j)
            res = (pr[0] // 2 ** j, pr[1] // 2 ** j)
            self.concat_back_dim.append(nn.Linear(2 * dim, dim) if i > 0 else nn.Identity())
            if i == 0:
                self.layers_up.append(PatchExpand(res, dim, 2))
            else:
                self.layers_up.append(BasicLayer(
                    dim, res, depths[j], num_heads[j], window_size, mlp_ratio, qkv_bias,
                    drop_rate, attn_drop_rate, dpr[sum(depths[:j]):sum(depths[:j + 1])],
                    upsample=PatchExpand if i < L - 1 else None))

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (L - 1)))
        self.norm_up = nn.LayerNorm(embed_dim)
        self.up = FinalPatchExpandX4((img_size // 4, img_size // 4), embed_dim)
        self.output = nn.Conv2d(embed_dim, num_classes, 1, bias=False)
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

    def forward(self, x):
        B = x.shape[0]
        x = self.pos_drop(self.patch_embed(x))
        skips = []
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        x = self.norm(x)
        for i, layer_up in enumerate(self.layers_up):
            if i > 0:
                x = self.concat_back_dim[i](torch.cat([x, skips[self.num_layers - 1 - i]], -1))
            x = layer_up(x)
        x = self.up(self.norm_up(x))
        H, W = self.up.input_resolution
        x = x.view(B, 4 * H, 4 * W, -1).permute(0, 3, 1, 2)
        return self.output(x)  # logits (B, num_classes, H, W)

    def load_pretrained(self, path_or_url=SWIN_TINY_URL):
        """Nạp trọng số Swin-T ImageNet cho bộ mã hóa VÀ sao chép đối xứng sang bộ giải mã
        (layers.k -> layers_up.(3-k)), đúng như Swin-Unet gốc. Các khóa lệch kích thước bị bỏ qua."""
        if str(path_or_url).startswith("http"):
            ckpt = torch.hub.load_state_dict_from_url(path_or_url, map_location="cpu", progress=True)
        else:
            ckpt = torch.load(path_or_url, map_location="cpu")
        pre = ckpt.get("model", ckpt)
        pre = {k: v for k, v in pre.items() if not k.startswith("head")}
        full = dict(pre)
        for k, v in pre.items():
            if k.startswith("layers."):
                full["layers_up." + str(3 - int(k[7])) + k[8:]] = v
        own = self.state_dict()
        loadable = {k: v for k, v in full.items() if k in own and own[k].shape == v.shape}
        self.load_state_dict(loadable, strict=False)
        print(f"[Swin-Unet] nạp {len(loadable)}/{len(own)} tensor từ trọng số tiền huấn luyện")
        return len(loadable)
