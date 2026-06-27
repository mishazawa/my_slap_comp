import OpenImageIO as oiio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from src.image import COLOR_PLANE, DEPTH_PLANE

from src.image_processing.crypto import (
    list_cryptopass,
    decode_cryptomatte,
)

from src.image_processing.filters import FilterConfig


def calc_depth(img, target_hash, sort_func=None):
    depth_plane = img.get_plane(DEPTH_PLANE)
    mask = decode_cryptomatte(img, target_hash)
    return sort_func(depth_plane["pixels"][mask > 0.0])


def slap_comp(img, sort_func, pass_processor, config: FilterConfig):
    crypto_passes = list_cryptopass(img)
    color_plane = img.get_plane(COLOR_PLANE)

    tasks = []

    for crypto_id, name, target_hash in crypto_passes:
        avg_depth = calc_depth(img, target_hash, sort_func)
        tasks.append((avg_depth, crypto_id, name, target_hash))

    tasks.sort(key=lambda x: x[0], reverse=True)

    height, width, _ = color_plane["pixels"].shape
    spec = oiio.ImageSpec(width, height, 4, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"]

    accumulated_buffer = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.zero(accumulated_buffer)

    preset = partial(
        pass_processor,
        img=img,
        color_plane=color_plane,
        config=config,
    )

    processed_layers = []

    with ThreadPoolExecutor() as executor:
        processed_layers = list(executor.map(preset, tasks))

    for layer_buf in processed_layers:
        oiio.ImageBufAlgo.over(accumulated_buffer, layer_buf, accumulated_buffer)

    final_gamma_buffer = oiio.ImageBuf()
    oiio.ImageBufAlgo.colorconvert(
        final_gamma_buffer, accumulated_buffer, "linear", "sRGB"
    )

    return final_gamma_buffer
