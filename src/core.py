import OpenImageIO as oiio
from functools import partial

from src.image import COLOR_PLANE, ZINDEX_PLANE

from src.image_processing.crypto import (
    list_cryptopass,
    decode_cryptomatte,
)

from src.image_processing.filters import FilterConfig

from src.utils import max


def calc_depth(img, target_hash):
    depth_plane = img.get_plane(ZINDEX_PLANE)
    mask = decode_cryptomatte(img, target_hash)
    return max(depth_plane["pixels"][..., 0][mask > 0.0])


def slap_comp(img, pass_processor, config: FilterConfig):
    crypto_passes = list_cryptopass(img)
    color_plane = img.get_plane(COLOR_PLANE)

    tasks = []

    for crypto_id, name, target_hash in crypto_passes:
        avg_depth = calc_depth(img, target_hash)
        tasks.append((avg_depth, crypto_id, name, target_hash))

    tasks.sort(key=lambda x: x[0])

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

    processed_layers = [preset(task) for task in tasks]

    for layer_buf in processed_layers:
        oiio.ImageBufAlgo.over(accumulated_buffer, layer_buf, accumulated_buffer)

    final_gamma_buffer = oiio.ImageBuf()
    oiio.ImageBufAlgo.colorconvert(
        final_gamma_buffer, accumulated_buffer, "linear", "sRGB"
    )

    return final_gamma_buffer
