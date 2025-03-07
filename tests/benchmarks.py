import triton
import torch

from functools import partial
from wan.modules.attention import _tile, _untile
from wan.modules.sta_flex_attn import get_sliding_tile_attention_mask
from wan.modules.natten_flex_attn import get_natten_mask
from torch.nn.attention.flex_attention import flex_attention, noop_mask, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)


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



F, H, W, NUM_HEADS, HEAD_DIM = 21, 30, 52, 16, 128
img_size = (F, H, W)
DEV = 'cuda'

sta_kernel_size = (3, 3, 7)
sta_tile_size = (3, 5, 4)

nattem_kernel_size = (9, 15, 28)


noop_mask = create_block_mask(noop_mask,
                             B=None,
                             H=None,
                             Q_LEN=F * H * W,
                             KV_LEN=F * H * W,
                             device=DEV,
                             _compile=True)

sta_mask = get_sliding_tile_attention_mask(
    kernel_size=sta_kernel_size,
    tile_size=sta_tile_size,
    img_size=img_size,
    device=DEV
)

natten_mask = get_natten_mask(
    kernel_size=nattem_kernel_size,
    img_size=img_size,
    device=DEV
)

def main():
    x = torch.randn(
        (1, F * W * H, NUM_HEADS, HEAD_DIM),
        device=DEV,
        dtype=torch.bfloat16
    )
    x_t = x.transpose(1, 2)

    if FLASH_ATTN_2_AVAILABLE:
        fla_v2_time = triton.testing.do_bench(
            lambda : flash_attn.flash_attn_func(
                q=x,
                k=x,
                v=x
            )
        )
    else:
        fla_v2_time = "invalid"

    if FLASH_ATTN_3_AVAILABLE:
        fla_v3_time = triton.testing.do_bench(
            lambda : flash_attn_interface.flash_attn_func(
                q=x,
                k=x,
                v=x
            )
        )
    else:
        fla_v3_time = "invalid"

    spda_time = triton.testing.do_bench(
        lambda : torch.nn.functional.scaled_dot_product_attention(
            query=x_t,
            key=x_t,
            value=x_t
        )
    )

    # JIT compile
    flex_attention(
                query=x_t,
                key=x_t,
                value=x_t,
                block_mask=noop_mask
            )
    flex_time = triton.testing.do_bench(
        lambda : flex_attention(
            query=x_t,
            key=x_t,
            value=x_t,
            block_mask=noop_mask
        )
    )

    global _tile
    _tile = partial(
        _tile,
        latent_size=img_size,
        tile_size=sta_tile_size
    )
    global _untile
    _untile = partial(
        _untile,
        img_size=img_size,
        tile_size=sta_tile_size
    )
    # JIT
    _untile(flex_attention(
            query=_tile(x),
            key=_tile(x),
            value=_tile(x),
            block_mask=sta_mask
        )
    )
    sta_time = triton.testing.do_bench(
        lambda : _untile(flex_attention(
            query=_tile(x),
            key=_tile(x),
            value=_tile(x),
            block_mask=sta_mask
        ))
    )

    # JIT
    flex_attention(
        query=x_t,
        key=x_t,
        value=x_t,
        block_mask=natten_mask
    )
    natten_time = triton.testing.do_bench(
        lambda : flex_attention(
            query=x_t,
            key=x_t,
            value=x_t,
            block_mask=natten_mask
        )
    )

    print(f"""
Flash Attention V2 Time: {fla_v2_time}
Flash Attention V3 Time: {fla_v3_time}
SPDA Time: {spda_time}
Flex Attention Time: {flex_time}
Sliding Tile Attention Time: {sta_time}
NATTEN time: {natten_time}
""")


if __name__ == '__main__':
    main()