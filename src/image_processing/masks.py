import numpy as np
import OpenImageIO as oiio

from .crypto import decode_cryptomatte


def create_mask_buf(img, target_hash):
    raw_mask = decode_cryptomatte(img, target_hash)

    height, width = raw_mask.shape[:2]
    mask_rgba = np.zeros((height, width, 4), dtype=np.float32)

    mask_bool = raw_mask > 0
    mask_rgba[mask_bool] = 1.0

    return oiio.ImageBuf(mask_rgba)


def get_masked_pixels(
    color_buf: oiio.ImageBuf, mask_buf: oiio.ImageBuf
) -> oiio.ImageBuf:

    oiio.ImageBufAlgo.mul(color_buf, color_buf, mask_buf)
    return color_buf
