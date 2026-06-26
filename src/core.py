import numpy as np
import OpenImageIO as oiio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from src.image import COLOR_PLANE, DEPTH_PLANE
from src.ops import (
    add_outline,
    add_shadow,
    apply_paper,
    smooth_mask,
)
from src.utils import get_masked_pixels, median, ensure_rgba_buf

from src.crypto import (
    list_cryptopass,
    decode_cryptomatte,
)
from src.settings import EXECUTOR_THREAD, EXECUTOR_SEQUENCE


def create_mask_buf(img, crypto_id, target_hash):
    raw_mask = decode_cryptomatte(img, crypto_id, target_hash)

    height, width = raw_mask.shape[:2]
    mask_rgba = np.zeros((height, width, 4), dtype=np.float32)

    mask_bool = raw_mask > 0
    mask_rgba[mask_bool] = 1.0

    return oiio.ImageBuf(mask_rgba)


def process_pass(
    img,
    color_plane,
    crypto_id,
    target_hash,
    mask_smooth_width,
    mask_smooth_height,
    outline_thickness,
    outline_color,
    shadow_color,
    shadow_intensity,
    light_vector,
):

    mask = create_mask_buf(img, crypto_id, target_hash)
    mask = smooth_mask(mask, mask_smooth_width, mask_smooth_height)

    a = ensure_rgba_buf(color_plane["pixels"])
    a = get_masked_pixels(a, mask)
    a = get_masked_pixels(apply_paper(a), mask)
    a = add_outline(a, outline_thickness)
    a = add_shadow(
        a,
        shadow_color=shadow_color,
        shadow_intensity=shadow_intensity,
        light_vector=light_vector,
    )

    return a


def _run_pass_helper(task_tuple, img, color_plane, **kwargs):
    _, crypto_id, name, target_hash = task_tuple
    return process_pass(img, color_plane, crypto_id, target_hash, **kwargs)


def slap_comp(img, executor_type=EXECUTOR_THREAD, **kwargs):
    crypto_passes = list_cryptopass(img)

    color_plane = img.get_plane(COLOR_PLANE)
    depth_plane = img.get_plane(DEPTH_PLANE)

    tasks = []

    for crypto_id, name, target_hash in crypto_passes:
        mask = decode_cryptomatte(img, crypto_id, target_hash)
        avg_depth = median(depth_plane["pixels"][mask > 0.0])
        tasks.append((avg_depth, crypto_id, name, target_hash))

    tasks.sort(key=lambda x: x[0], reverse=True)

    height, width, _ = color_plane["pixels"].shape
    spec = oiio.ImageSpec(width, height, 4, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"]

    accumulated_buffer = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.zero(accumulated_buffer)

    run_pass_func = partial(
        _run_pass_helper,
        img=img,
        color_plane=color_plane,
        **kwargs,
    )

    processed_layers = []

    if executor_type == EXECUTOR_SEQUENCE:
        processed_layers = [run_pass_func(k) for k in tasks]

    if executor_type == EXECUTOR_THREAD:
        with ThreadPoolExecutor() as executor:
            processed_layers = list(executor.map(run_pass_func, tasks))

    for layer_buf in processed_layers:
        oiio.ImageBufAlgo.over(accumulated_buffer, layer_buf, accumulated_buffer)

    final_gamma_buffer = oiio.ImageBuf()
    oiio.ImageBufAlgo.colorconvert(
        final_gamma_buffer, accumulated_buffer, "linear", "sRGB"
    )

    return final_gamma_buffer
