import logging
from typing import Union

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """Conditional UNet.

    local_cond_dim: dimension of local condition. local_cond is encoded with a
        residual block along with the diffusion timestep and added to the output
        of the first down block. It's also encoded with a separate residual
        block (also along with the diffusion timestep) and added to the output
        of the last up block.
    global_cond_dim: dimension of global condition. It's concatenated to the
        diffusion timestep and used as a global condition for all residual blocks.
    """
    def __init__(self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor], #, float, int
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # print("sample:", sample.shape)
        sample = einops.rearrange(sample, 'b h t -> b t h')
        # print("sample:", sample.shape)
        # print("timestep:", timestep.shape)
        # if local_cond is not None:
        #     print("local_cond:", local_cond.shape)
        # if global_cond is not None:
        #     print("global_cond:", global_cond.shape)

        # 1. time
        timesteps = timestep
        ### Our current framework does not need any of this,
        ### We always pass a tensor, with len(timesteps.shape) == 1
        # print(timestep.shape, type(timestep), len(timestep.shape),
        #       not torch.is_tensor(timesteps), torch.is_tensor(timesteps) and len(timesteps.shape) == 0)
        # if not torch.is_tensor(timesteps):
        #     # TO DO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        #     timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            # Concatenate the global conditioning to the diffusion timestep.
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        # encode local features with two separate residual blocks along with the
        # diffusion timestep.
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                # Add the local condition to the output of the first down block.
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            # print("idx:", idx)
            # print("x:", x.shape)
            # print("h:", [h_i.shape for h_i in h])
            h_i = h.pop()
            # print(f"h_i: {h_i.shape}    x: {x.shape}")
            x = torch.cat((x, h_i), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            # The original code is:
            # if idx == len(self.up_modules) and len(h_local) > 0:
            if idx == (len(self.up_modules)-1) and len(h_local) > 0:
                # Add the local condition to the output of the last up block.
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
