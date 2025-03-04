# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

from einops import rearrange
from typing import Tuple

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out

__OPERAND_1 = "b (n_f t_f n_h t_h n_w t_w) n_head head_dim"
__OPERAND_2 = "b n_head (n_f n_h n_w t_f t_h t_w) head_dim"

def _tile(x: torch.Tensor, latent_size: Tuple[int, int, int], tile_size: Tuple[int, int, int]) -> torch.Tensor:
    """
        Flatten strategy for Wan
        Reference:
            https://github.com/hao-ai-lab/FastVideo/blob/554ee17de54b95432edd4465a65e75d809b4564f/fastvideo/models/hunyuan/modules/attenion.py#L37
    """
    for L, l in zip(latent_size, tile_size):
        if L % l != 0:
            raise ValueError(f"Tile size must divide video latent, found {L=} {l=}, {latent_size=}, try adjusing frame number, resolution or tile_size")

    img_f, img_h, img_w = latent_size
    tile_f, tile_h, tile_w = tile_size
    x = rearrange(
        x,
        __OPERAND_1 + " -> " + __OPERAND_2,
        n_f=img_f // tile_f,
        n_h=img_h // tile_h,
        n_w=img_w // tile_w,
        t_f=tile_f,
        t_h=tile_h,
        t_w=tile_w
    )
    return x


def _untile(x: torch.Tensor, img_size: Tuple[int, int, int], tile_size: Tuple[int, int, int]) -> torch.Tensor:
    for L, l in zip(img_size, tile_size):
        if L % l != 0:
            raise ValueError("Tile size must divide video latent")

    img_f, img_h, img_w = img_size
    tile_f, tile_h, tile_w = tile_size
    x = rearrange(
        x,
        __OPERAND_2 + " -> " + __OPERAND_1,
        n_f=img_f // tile_f,
        n_h=img_h // tile_h,
        n_w=img_w // tile_w,
        t_f=tile_f,
        t_h=tile_h,
        t_w=tile_w
    )
    return x

DEFAULT_TILE_SIZE = (6, 8, 8)
DEBUG = True

def sliding_tile_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Tuple[int, int, int],
    latent_size: Tuple[int, int, int],
    tile_size: Tuple[int, int, int] = DEFAULT_TILE_SIZE
):
    """
    Args:
        q:              [B, Lq, Nq, C1]. batch, seqlen, head_num, head_dim
        k:              [B, Lk, Nk, C1].
        v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
        window_size:    STA window size, each tile is (6, 8, 8) as stated in wan/csrc/sliding_tile_attention/README.md
        img_size:       [frame, height, width]
    """
    TK_IMPL_AVAILABLE = True
    try:
        from st_attn import sliding_tile_attention
    except ImportError as e:
        TK_IMPL_AVAILABLE = False
        print("Could not load cuda Sliding Tile Attention, using Flex Attention instead")
        from .sta_flex_attn import get_sliding_tile_attention_mask
        from torch.nn.attention.flex_attention import flex_attention

    # TODO @botbw: better way of doing this
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    _, _, head_num, _ = q.shape
    # tile inputs
    q = _tile(x=q, latent_size=latent_size, tile_size=tile_size)
    k = _tile(x=k, latent_size=latent_size, tile_size=tile_size)
    v = _tile(x=v, latent_size=latent_size, tile_size=tile_size)

    if TK_IMPL_AVAILABLE:
        assert tile_size == DEFAULT_TILE_SIZE, "TK impl only supports default tile size"
        o = sliding_tile_attention(
            q_all=q,
            k_all=k,
            v_all=v,
            window_size=[window_size] * head_num,  # TODO @botbw: support different window for different attn head and searching strategy
            text_length=0,
            has_text=False
        )
    else:
        block_mod = get_sliding_tile_attention_mask(
            kernel_size=window_size,
            tile_size=tile_size,
            img_size=latent_size,
            text_length=0,
            device=q.device,
            text_max_len=0
        )
        if DEBUG:
            from attn_gym import visualize_attention_scores
            # install from https://github.com/pytorch-labs/attention-gym/tree/main
            visualize_attention_scores(
                query=q,
                key=k,
                mask_mod=block_mod.mask_mod,
                name='flex-attn-visual'
            )
        o = flex_attention(
            query=q,
            key=k,
            value=v,
            block_mask=block_mod
        )

    return _untile(x=o, img_size=latent_size, tile_size=tile_size)
