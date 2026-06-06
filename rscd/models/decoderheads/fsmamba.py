import torch
import torch.nn as nn
from rscd.models.backbones.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute, SS2D, Mlp, AttentionGuidedVSSBlock
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from functools import partial
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from typing import Optional, Callable, Any, Tuple
def detach_feature(x):
    if x is None:
        return None
    return x.detach()


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.output_dim = dim * (dim_scale ** 2)
        self.expand = nn.Linear(dim, self.output_dim, bias=False)
        self.norm = norm_layer(self.output_dim // (dim_scale ** 2))

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"Input channels ({C}) do not match initialized channels ({self.dim}).")
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.expand.in_features != C:
            self.expand = nn.Linear(C, self.output_dim, bias=False).to(x.device)
        x = self.expand(x)
        if self.dim_scale != 1:
            group_factor = self.dim_scale ** 2
            c = self.output_dim // group_factor
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                          p1=self.dim_scale, p2=self.dim_scale, c=c)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class UpsampleExpand(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.output_dim = dim // 2
        self.linear = nn.Linear(dim, self.output_dim, bias=False)
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"Input channels ({C}) do not match initialized channels ({self.dim}).")
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.linear.in_features != C:
            self.linear = nn.Linear(C, self.output_dim, bias=False).to(x.device)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class FinalUpsample_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.dim:
            raise ValueError(f"Input channels ({C}) do not match initialized channels ({self.dim}).")
        x = x.permute(0, 2, 3, 1).contiguous()
        if self.linear1.in_features != C:
            self.linear1 = nn.Linear(C, self.linear1.out_features, bias=False).to(x.device)
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if self.linear2.in_features != C:
            self.linear2 = nn.Linear(C, self.linear2.out_features, bias=False).to(x.device)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class GlobalLocalCollaborativeDecoderStage(nn.Module):
    def __init__(self, dim, input_resolution, depth, dt_rank="auto",
                 d_state=4, ssm_ratio=2.0, attn_drop_rate=0.,
                 drop_rate=0.0, mlp_ratio=4.0,
                 drop_path=0.1, norm_layer=nn.LayerNorm, upsample=None,
                 shared_ssm=False, softmax_version=False,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        if not isinstance(drop_path, list):
            drop_path = [drop_path] * depth
        self.blocks = nn.ModuleList([
            GlobalLocalCollaborativeBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if i < len(drop_path) else drop_path[0],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
            )
            for i in range(depth)])
        if upsample is not None:
            self.upsample = UpsampleExpand(input_resolution, dim=dim, patch_size=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = avg_out + max_out
        return x * self.sigmoid(attn)


class ChannelAttentionBlock(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(ChannelAttentionBlock, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class GlobalLocalCollaborativeBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            drop_path: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            d_state: int = 16,
            dt_rank: int = "auto",
            ssm_ratio: float = 2.0,
            attn_drop_rate: float = 0.0,
            shared_ssm: bool = False,
            softmax_version: bool = False,
            use_checkpoint: bool = False,
            reduction_ratio: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.hidden_dim = hidden_dim
        self.reduction_ratio = reduction_ratio
        if not isinstance(drop_path, list):
            drop_path = [drop_path]
        self.norm = LayerNorm2d(hidden_dim)
        self.op = AttentionGuidedVSSBlock(
            hidden_dim=hidden_dim,
            drop_path=drop_path[0] if len(drop_path) > 0 else drop_path,
            norm_layer=LayerNorm2d,
            channel_first=True,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=attn_drop_rate,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=0.0,
            use_checkpoint=use_checkpoint,
            post_norm=False,
        )
        self.feature_scale = nn.Parameter(torch.ones(hidden_dim))
        dp = drop_path[0] if len(drop_path) > 0 else drop_path
        self.drop_path = DropPath(dp) if dp > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        identity = x
        feat = self.op(self.norm(x))
        feat = feat * self.feature_scale.view(1, -1, 1, 1)
        return identity + self.drop_path(feat)


class GlobalLocalCollaborativeFusion(nn.Module):
    """Fuse FDEM and STPM features following the GLCD formulation.

    The module implements the paper-level flow:
    concat(F_FDEM, F_STPM) -> 1x1 projection Z, a global SS2D/VSS branch,
    a local multi-kernel convolution branch, and branch-wise Softmax fusion.
    """

    def __init__(self, channels, drop_path=0.1):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.project = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.global_branch = VSSBlock(
            hidden_dim=channels,
            drop_path=drop_path,
            norm_layer=LayerNorm2d,
            channel_first=True,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=0.0,
            use_checkpoint=False,
            post_norm=False,
        )
        self.local_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, bias=False),
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, bias=False),
        ])
        self.local_project = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
        self.branch_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, fdem_feature, stpm_feature):
        z = self.project(torch.cat([fdem_feature, stpm_feature], dim=1))
        global_feature = self.global_branch(z)
        local_feature = self.local_project(torch.cat([conv(z) for conv in self.local_convs], dim=1))
        weights = self.branch_attention(torch.cat([global_feature, local_feature], dim=1))
        return weights[:, 0:1] * global_feature + weights[:, 1:2] * local_feature


class AdaptiveFrequencyAttention(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFrequencyAttention, self).__init__()
        self.in_channels = in_channels
        self.freq_conv = nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 1, bias=False),
            nn.Softmax(dim=1)
        )
        self.low_freq = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.mid_freq = nn.Conv2d(in_channels, in_channels, 5, padding=2, bias=False)
        self.high_freq = nn.Conv2d(in_channels, in_channels, 7, padding=3, bias=False)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        freq_feat = self.freq_conv(x)
        weights = self.weight_generator(freq_feat)
        low_response = self.low_freq(freq_feat)
        mid_response = self.mid_freq(freq_feat)
        high_response = self.high_freq(freq_feat)
        b, c, h, w = low_response.shape
        weight1 = weights[:, 0:1, :, :].expand(b, c, h, w)
        weight2 = weights[:, 1:2, :, :].expand(b, c, h, w)
        weight3 = weights[:, 2:3, :, :].expand(b, c, h, w)
        combined = (weight1 * low_response +
                    weight2 * mid_response +
                    weight3 * high_response)
        spatial_att = self.spatial_att(combined)
        return spatial_att


class EnhancedFourierUnit(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(EnhancedFourierUnit, self).__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, 1, 1, 0, groups=self.groups, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.PReLU(),
        )
        self.attention = AdaptiveFrequencyAttention(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ffted_x, ffted_y):
        batch, c, h, w = ffted_x.size()
        real_x, imag_x = ffted_x.real, ffted_x.imag
        real_y, imag_y = ffted_y.real, ffted_y.imag
        diff_real = real_x - real_y
        diff_imag = imag_x - imag_y
        ffted = torch.cat([diff_real, diff_imag], dim=1)
        ffted = self.conv_layer(ffted)
        attention_mask = self.attention(ffted)
        ffted = ffted * attention_mask
        processed_real, processed_imag = torch.chunk(ffted, 2, dim=1)
        processed_fft = torch.complex(processed_real, processed_imag)
        return processed_fft


class FourierUnit(nn.Module):
    def __init__(self, in_channel, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel * 2, 1, 1, 0, groups=self.groups, bias=False),
            nn.BatchNorm2d(in_channel * 2),
            nn.PReLU(),
        )
        self.conv7x7 = nn.Conv2d(2, 1, 7, padding=7 // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ffted_x, ffted_y):
        batch, c, h, w = ffted_x.size()
        real_x, imag_x = ffted_x.real, ffted_x.imag
        real_y, imag_y = ffted_y.real, ffted_y.imag
        diff_real = real_x - real_y
        diff_imag = imag_x - imag_y
        ffted = torch.cat([diff_real, diff_imag], dim=1)
        ffted = self.conv_layer(ffted)
        max_out, _ = torch.max(ffted, dim=1, keepdim=True)
        avg_out = torch.mean(ffted, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv7x7(torch.cat([max_out, avg_out], dim=1)))
        ffted = ffted * mask
        processed_real, processed_imag = torch.chunk(ffted, 2, dim=1)
        processed_fft = torch.complex(processed_real, processed_imag)
        return processed_fft


class MixingBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.convmix = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, 3, groups=in_channel, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.PReLU(),
        )

    def forward(self, x, y):
        mixed = torch.stack((x, y), dim=2)
        mixed = mixed.view(x.shape[0], -1, x.shape[2], x.shape[3])
        return self.convmix(mixed)


class SpatioTemporalPerceptionModule(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.mix = MixingBlock(in_channel)
        self.conv_sub = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.PReLU()
        )
        self.vss_block = VSSBlock(
            hidden_dim=in_channel,
            drop_path=0.1,
            norm_layer=LayerNorm2d,
            channel_first=True,
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            mlp_ratio=0.0,
            use_checkpoint=False,
            post_norm=False,
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.PReLU()
        )

        # 闁告瑯鍨甸～瀣礌閺嶎偆澶勯悗?
        self.stpm_before = None
        self.stpm_after = None

    def forward(self, x1, x2):
        x_sub = self.conv_sub(torch.abs(x1 - x2))
        x_mix = self.mix(x1, x2)
        x_fus = x_mix + x_mix.mul(x_sub)

        # STPM闁告挸绋勭槐鐗堟交濞戞ê寮砎SSBlock闁?
        self.stpm_before = detach_feature(x_fus)

        x_fus = self.vss_block(x_fus)
        x = self.conv_cat(x_fus)

        # STPM闁告艾鍑界槐鐧眔nv_cat閺夊牊鎸搁崵?
        self.stpm_after = detach_feature(x)

        return x



class FrequencyDomainEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.fourier_unit = EnhancedFourierUnit(in_channels)
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.PReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.output_refine = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.PReLU(),
            nn.Conv2d(in_channels * 2, in_channels * 2, 1, bias=False),
        )

        # Cached only for optional visualization/debugging.
        self.fdem_before = None
        self.fdem_after = None

    def forward(self, x_A, x_B):
        fdem_before = torch.abs(x_A - x_B)

        fft_A = torch.fft.fft2(x_A, dim=(-2, -1))
        fft_B = torch.fft.fft2(x_B, dim=(-2, -1))
        delta_fft = fft_A - fft_B

        processed_fft = self.fourier_unit(fft_A, fft_B)
        freq_att = self.freq_attention(torch.abs(processed_fft))
        processed_fft = processed_fft * freq_att

        refined_fft = processed_fft + delta_fft
        refined_fft = torch.cat([refined_fft.real, refined_fft.imag], dim=1)
        refined_fft = self.output_refine(refined_fft)
        refined_real, refined_imag = torch.chunk(refined_fft, 2, dim=1)
        processed_fft = torch.complex(refined_real, refined_imag)

        diff_spatial = torch.fft.ifft2(processed_fft, dim=(-2, -1)).real

        self.fdem_before = detach_feature(fdem_before)
        self.fdem_after = detach_feature(diff_spatial)

        return diff_spatial



class FSMamba(nn.Module):
    def __init__(self,
                 img_size: Optional[Tuple[int, int]] = None,
                 in_channels=[64, 128, 192, 256],
                 num_classes=2,
                 dims=96,
                 embed_dim=96,
                 align_corners=False,
                 patch_size=4,
                 depths=[4, 4, 4, 4],
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        # img_size 濞寸姴鎳嶇紞鏃堝础閻樿京绉撮柨娑樺缁楀宕ｉ崒娆戠憿閻庡湱鍋ゅ顖滄媼閿涘嫮鏆?
        base_h, base_w = (img_size if img_size is not None else (1, 1))
        self.patches_resolution = [base_h // patch_size, base_w // patch_size]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.frequency_domain_enhancement_modules = nn.ModuleList([
            FrequencyDomainEnhancementModule(in_channels[i]) for i in range(4)
        ])
        self.spatio_temporal_perception_modules = nn.ModuleList([
            SpatioTemporalPerceptionModule(in_channels[i]) for i in range(4)
        ])
        self.global_local_collaborative_fusions = nn.ModuleList([
            GlobalLocalCollaborativeFusion(in_channels[i], drop_path=drop_path_rate) for i in range(4)
        ])
        self.dim_adapters = nn.ModuleList([
            nn.Conv2d(in_channels[i], in_channels[i], 1, bias=False) for i in range(4)
        ])
        self.layers_up = nn.ModuleList()
        self.up = FinalUpsample_X4(
            input_resolution=(base_h // patch_size, base_w // patch_size),
            patch_size=4,
            dim=in_channels[0]
        )
        self.output = nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=num_classes,
            kernel_size=1,
            bias=False
        )
        self.in_channels = in_channels
        
        #xinjia#
        self.glcd_before = None
        self.glcd_after = None

        for i_layer in range(self.num_layers):
            ch_in = in_channels[self.num_layers - 1 - i_layer]
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=ch_in,
                    dim_scale=2,
                    norm_layer=norm_layer)
            else:
                layer_up = GlobalLocalCollaborativeDecoderStage(
                    dim=ch_in,
                    input_resolution=(
                        self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)

        # 濡澘瀚紓鎾舵崉鐎圭姷绠鹃梺顐ｅ哺娴滈箖鏌呴崒鐐插赋闁革綆鐓夌槐婵嬫焼閸喖甯抽柛?forward 濞戞搩鍘烘径宥夊籍鐠哄搫鐏＄€点倗灏ㄧ槐娆撳矗椤栨繍鍞茬紓浣稿暔閳ь兛绀佽ぐ鍙夌┍濠靛棛鎽犻柨?
        self.merge_adapters = nn.ModuleList()
        prev_ch = in_channels[-1]  # stage0 (PatchExpand) 閺夊牊鎸搁崵顓㈡焻濮樻湹澹曞ǎ鍥ㄧ箖鐎垫梹绋夊鍛秮
        for i in range(1, self.num_layers):
            out_ch = in_channels[-1 - i]  # 鐟滅増鎸告晶鐘典沪?x_res 闁汇劌瀚伴埀顒佸哺娴?
            if prev_ch != out_ch:
                self.merge_adapters.append(nn.Conv2d(prev_ch, out_ch, kernel_size=1, bias=False))
            else:
                self.merge_adapters.append(nn.Identity())
            # Non-final GLCD stages with UpsampleExpand halve the feature channels.
            if i < self.num_layers - 1:
                prev_ch = out_ch // 2
            else:
                prev_ch = out_ch

        self.norm_up = norm_layer(in_channels[0])
        # 闁活潿鍔嬬花鐝縭ad-CAM闁汇劌鍩卭ok
        self.fused_features = [None] * 4
        self.mamba_outputs = [None] * 4

    def forward_up_features(self, inputs):
        x = None
        y = None

    # GLCD闁告挸绋勭槐鐗堟交濞戞ê寮砫ecoder闁汇劌瀚〒鑸殿殗濡櫣婀撮柧璇茬Т閹酣鎮ч悷鎵獧
        self.glcd_before = detach_feature(inputs[3])

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = inputs[3 - inx]
                y = layer_up(x)
            else:
                x_res = inputs[3 - inx]
                y_up = F.interpolate(
                    y,
                    size=x_res.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                y_up = self.merge_adapters[inx - 1](y_up)
                x = y_up + x_res
                y = layer_up(x)

        x = y.permute(0, 2, 3, 1).contiguous()
        x = self.norm_up(x)
        x = x.permute(0, 3, 1, 2).contiguous()

    # GLCD闁告艾鍑界槐鐧瞖coder缂備焦鎸诲顐﹀触鎼达絾鐣遍柣妤冩嚀缁?
        self.glcd_after = detach_feature(x)

        return x


    def forward(self, inputs):
        x_A, x_B = inputs
        outs_fused = []
        outs_A = x_A
        outs_B = x_B
        for i in range(4):
            out_A = outs_A[i]
            out_B = outs_B[i]
            fdem_feature = self.frequency_domain_enhancement_modules[i](out_A, out_B)
            stpm_feature = self.spatio_temporal_perception_modules[i](out_A, out_B)
            fused_feature = self.global_local_collaborative_fusions[i](fdem_feature, stpm_feature)
            fused_feature = self.dim_adapters[i](fused_feature)
            outs_fused.append(fused_feature)
        # 濞ｅ洦绻傞悺銊╂懚瀹ュ懏鍊ら柛姘捣濞堟垿鎮ч悷鎵獧闁活潿鍔嬬花鐝縭ad-CAM
        self.fused_features = outs_fused
        x = self.forward_up_features(outs_fused)  # [B, C, H, W]
        x = self.up(x)                            # [B, C, 4H, 4W]
        x_last = self.output(x)                   # [B, num_classes, 4H, 4W]
        return x_last
