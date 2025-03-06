
from typing import Tuple

import torch
from functools import cache
from torch import BoolTensor, IntTensor
from torch.nn.attention.flex_attention import create_block_mask

torch._inductor.config.realize_opcount_threshold = 100
torch._dynamo.config.cache_size_limit = 1000

def generate_natten(
    img_size: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
):
    """Generates a NATTEN attention mask with a given kernel size.
    Args:
        canvas_w: The width of the canvas.
        canvas_h: The height of the canvas.
        kernel_w: The width of the kernel.
        kernel_h: The height of the kernel.
    """
    img_t, img_h, img_w = img_size
    k_t, k_h, k_w = kernel_size

    def get_x_y(idx):
        t = idx // (img_h * img_w)
        h = (idx % (img_h * img_w)) // img_w
        w = idx % img_w
        return t, h, w

    def natten_mask_mod(
        b,
        h,
        q_idx,
        kv_idx,
    ):
        q_t, q_h, q_w = get_x_y(q_idx)
        kv_t, kv_h, kv_w = get_x_y(kv_idx)
        # kernel nominally attempts to center itself on the query, but kernel center
        # is clamped to a fixed distance (kernel half-length) from the canvas edge
        kernel_center_t = q_t.clamp(k_t // 2, (img_t - 1) - k_t // 2)
        kernel_center_h = q_h.clamp(k_h // 2, (img_h - 1) - k_h // 2)
        kernel_center_w = q_w.clamp(k_w // 2, (img_w - 1) - k_w // 2)
        t_mask = (kernel_center_t - kv_t).abs() <= k_t // 2
        h_mask = (kernel_center_h - kv_h).abs() <= k_h // 2
        w_mask = (kernel_center_w - kv_w).abs() <= k_w // 2
        return t_mask & h_mask & w_mask

    natten_mask_mod.__name__ = f"natten_{img_size}_{kernel_size}"
    return natten_mask_mod


@cache
def _get_natten_mask(kernel_size, img_size, device):
    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    image_mask = generate_natten(img_size, kernel_size)
    mask = create_block_mask(image_mask,
                             B=None,
                             H=None,
                             Q_LEN=img_seq_len,
                             KV_LEN=img_seq_len,
                             device=device,
                             _compile=True)
    return mask


def get_natten_mask(kernel_size, img_size, device):
    return _get_natten_mask(kernel_size, img_size, str(device))
