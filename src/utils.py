import OpenImageIO as oiio
import numpy as np
from src.image import Image
import struct


def read_image(filepath):
    """
    Reads an EXR image into an Image object.
    """
    return Image.read(filepath)


def write_image(filepath, image):
    """
    Writes an Image object to a file.
    """
    plane_name = next(iter(image.subimages))
    plane = image.get_plane(plane_name)

    height, width, channels = plane["pixels"].shape
    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    spec.channelnames = plane["channel_names"]

    output_file = oiio.ImageOutput.create(filepath)
    if not output_file:
        raise RuntimeError(f"Could not create output file: {filepath}")

    output_file.open(filepath, spec)
    output_file.write_image(plane["pixels"])
    output_file.close()


def list_image_planes(image):
    """
    Returns a list of names of subimages (planes) in the Image object.
    """
    return list(image.subimages.keys())


def min(data):
    return np.min(data)


def max(data):
    return np.max(data)


def average(data):
    return np.mean(data)


def normalize(data):
    d_min = np.min(data)
    d_max = np.max(data)
    if d_max > d_min:
        return (data - d_min) / (d_max - d_min)
    return data


def hex_to_float32(hex_str: str) -> float:
    """Converts the Cryptomatte manifest hex string to the correct float32."""
    # Convert hex string to uint32 integer
    uint_val = int(hex_str, 16)

    # Cryptomatte NaN protection
    if (uint_val & 0x7E000000) == 0x7E000000:
        uint_val = (uint_val & 0x007FFFFF) | 0x3F000000

    # Bitcast using '>I' and '>f' (Big-endian / Network byte order)
    packed = struct.pack(">I", uint_val)
    return struct.unpack(">f", packed)[0]


def get_masked_pixels(pixels, mask):
    """
    Returns an RGBA image where the masked region is extracted from the input pixels.
    """
    height, width, _ = pixels.shape
    masked = np.zeros((height, width, 4), dtype=np.float32)
    mask_bool = mask > 0
    masked[mask_bool, :3] = pixels[mask_bool, :3]
    masked[mask_bool, 3] = 1.0
    return masked


def add_shadow_to_a_layer(
    pixels,
    offset=(15, 15),
    blur_radius=15.0,
    shadow_color=(0.0, 0.0, 0.0),
    shadow_intensity=0.5,
):
    """Adds a soft customizable shadow to a layer using OpenImageIO ImageBufAlgo.

    Parameters:
    - pixels (np.ndarray): Input image array (RGB or RGBA).
    - offset (tuple): (y_offset, x_offset) in pixels to shift the shadow.
    - blur_radius (float): Softness of the shadow.
    - shadow_color (tuple): (R, G, B) normalized color of the shadow (e.g., (0,0,0) for black).
    - shadow_intensity (float): Max opacity of the shadow (0.0 to 1.0).
    """
    height, width, channels = pixels.shape

    if channels == 3:
        alpha = np.ones((height, width, 1), dtype=np.float32)
        fg_pixels = np.concatenate([pixels, alpha], axis=2)
    else:
        fg_pixels = pixels.astype(np.float32)
        alpha = fg_pixels[:, :, 3:4]

    spec = oiio.ImageSpec(width, height, 4, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B", "A"]

    fg_buf = oiio.ImageBuf(spec)
    fg_buf.set_pixels(oiio.ROI(0, width, 0, height), fg_pixels)

    y_off, x_off = offset
    shifted_alpha = np.roll(alpha, shift=(y_off, x_off), axis=(0, 1))

    if y_off > 0:
        shifted_alpha[:y_off, :] = 0
    elif y_off < 0:
        shifted_alpha[y_off:, :] = 0

    if x_off > 0:
        shifted_alpha[:, :x_off] = 0
    elif x_off < 0:
        shifted_alpha[:, x_off:] = 0

    rgba_mask_pixels = np.concatenate([shifted_alpha] * 4, axis=2)

    shadow_mask_buf = oiio.ImageBuf(spec)
    shadow_mask_buf.set_pixels(oiio.ROI(0, width, 0, height), rgba_mask_pixels)

    if blur_radius > 0:
        K = oiio.ImageBufAlgo.make_kernel("gaussian", blur_radius, blur_radius)
        blurred_shadow_mask = oiio.ImageBuf(spec)
        oiio.ImageBufAlgo.convolve(blurred_shadow_mask, shadow_mask_buf, K)
    else:
        blurred_shadow_mask = shadow_mask_buf

    shadow_color_buf = oiio.ImageBuf(spec)

    fill_color = (
        shadow_color[0],
        shadow_color[1],
        shadow_color[2],
        shadow_intensity,
    )
    oiio.ImageBufAlgo.fill(shadow_color_buf, fill_color)

    shadow_final = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.mul(shadow_final, shadow_color_buf, blurred_shadow_mask)

    final_composite = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.over(final_composite, shadow_final, final_composite)
    oiio.ImageBufAlgo.over(final_composite, fg_buf, final_composite)

    return final_composite.get_pixels(oiio.FLOAT)
