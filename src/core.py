import numpy as np
import OpenImageIO as oiio

from src.image import COLOR_PLANE, DEPTH_PLANE
from src.utils import add_shadow_to_a_layer, get_masked_pixels, average
from src.crypto import (
    list_cryptopass,
    decode_cryptomatte,
)


def slap_comp(img):
    crypto_passes = list_cryptopass(img)

    color_plane = img.get_plane(COLOR_PLANE)
    depth_plane = img.get_plane(DEPTH_PLANE)

    ordered_passes_pool = []

    for crypto_id, name, target_hash in crypto_passes:
        if name == "/ground/mesh_0":
            ordered_passes_pool.append((0.0, crypto_id, name, target_hash))
            continue

        mask = decode_cryptomatte(img, crypto_id, target_hash)
        avg_depth = average(depth_plane["pixels"][mask > 0.0])
        ordered_passes_pool.append((avg_depth, crypto_id, name, target_hash))

    ordered_passes_pool.sort(key=lambda x: x[0])

    height, width, channels = color_plane["pixels"].shape
    spec = oiio.ImageSpec(width, height, 4, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"]

    base_pixels = np.zeros((height, width, 4), dtype=np.float32)

    accumulated_buffer = oiio.ImageBuf(spec)
    accumulated_buffer.set_pixels(
        oiio.ROI(0, width, 0, height), base_pixels.astype(np.float32)
    )

    for i, k in enumerate(ordered_passes_pool):
        avg_depth, crypto_id, name, target_hash = k
        print(f"Compositing: {name} (Avg Depth: {avg_depth:.2f})")
        print(f"Processing pass: {crypto_id} -> {name} with hash: {target_hash}")
        mask = decode_cryptomatte(img, crypto_id, target_hash)
        layer = get_masked_pixels(color_plane["pixels"], mask)
        shadowed_pixels = add_shadow_to_a_layer(
            layer,
            offset=(-4, -4),
            blur_radius=5.0,
            shadow_color=(0.0, 0.0, 0.0),
            shadow_intensity=0.7,
        )

        layer_buf = oiio.ImageBuf(spec)
        layer_buf.set_pixels(oiio.ROI(0, width, 0, height), shadowed_pixels)
        oiio.ImageBufAlgo.over(accumulated_buffer, layer_buf, accumulated_buffer)

    final_gamma_buffer = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.colorconvert(
        final_gamma_buffer, accumulated_buffer, "linear", "sRGB"
    )
    return final_gamma_buffer
