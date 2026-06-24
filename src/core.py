import numpy as np
import OpenImageIO as oiio

from src.image import COLOR_PLANE, DEPTH_PLANE
from src.ops import (
    add_shadow_to_a_layer,
    add_outline_to_layer,
    apply_paper,
    smooth_mask,
)
from src.utils import (
    get_masked_pixels,
    average,
)
from settings import (
    SHADOW_COLOR,
    SHADOW_INTENSITY,
    MASK_SMOOTH_WIDTH,
    MASK_SMOOTH_HEIGHT,
    OUTLINE_THICKNESS,
    OUTLINE_COLOR,
)
from src.crypto import (
    list_cryptopass,
    decode_cryptomatte,
)


def slap_comp(img):
    crypto_passes = list_cryptopass(img)

    color_plane = img.get_plane(COLOR_PLANE)
    depth_plane = img.get_plane(DEPTH_PLANE)

    ordered_passes_pool = []
    static_passes_pool = []

    for crypto_id, name, target_hash in crypto_passes:
        if name == "/ground/mesh_0":
            static_passes_pool.append((0.0, crypto_id, name, target_hash))
            continue

        mask = decode_cryptomatte(img, crypto_id, target_hash)
        avg_depth = average(depth_plane["pixels"][mask > 0.0])
        ordered_passes_pool.append((avg_depth, crypto_id, name, target_hash))

    ordered_passes_pool.sort(key=lambda x: x[0], reverse=True)

    height, width, channels = color_plane["pixels"].shape
    spec = oiio.ImageSpec(width, height, 4, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"]

    base_pixels = np.zeros((height, width, 4), dtype=np.float32)

    accumulated_buffer = oiio.ImageBuf(spec)
    accumulated_buffer.set_pixels(
        oiio.ROI(0, width, 0, height), base_pixels.astype(np.float32)
    )

    for i, k in enumerate(static_passes_pool + ordered_passes_pool):
        avg_depth, crypto_id, name, target_hash = k
        print(f"Compositing: {name} (Avg Depth: {avg_depth:.2f})")
        print(f"Processing pass: {crypto_id} -> {name} with hash: {target_hash}")

        mask = smooth_mask(
            decode_cryptomatte(img, crypto_id, target_hash),
            MASK_SMOOTH_WIDTH,
            MASK_SMOOTH_HEIGHT,
        )

        layer = get_masked_pixels(color_plane["pixels"], mask)

        layer = apply_paper(layer)

        outlined_layer = add_outline_to_layer(
            layer,
            outline_thickness=OUTLINE_THICKNESS,
            outline_color=OUTLINE_COLOR,
        )

        shadowed_pixels = add_shadow_to_a_layer(
            outlined_layer,
            shadow_color=SHADOW_COLOR,
            shadow_intensity=SHADOW_INTENSITY,
        )

        layer_buf = oiio.ImageBuf(spec)
        layer_buf.set_pixels(oiio.ROI(0, width, 0, height), shadowed_pixels)
        oiio.ImageBufAlgo.over(accumulated_buffer, layer_buf, accumulated_buffer)

    final_gamma_buffer = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.colorconvert(
        final_gamma_buffer, accumulated_buffer, "linear", "sRGB"
    )
    return final_gamma_buffer
