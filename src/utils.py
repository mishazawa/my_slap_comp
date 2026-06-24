import struct
import OpenImageIO as oiio
import numpy as np
from src.image import Image


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


def desaturate_pixels(pixels, factor=0.15):
    """
    Slightly desaturates an RGB or RGBA image. Safe for single-channel inputs.
    """
    # 1. Check channels
    channels = pixels.shape[2] if pixels.ndim == 3 else 1

    # If the image is already single-channel (grayscale), it cannot be desaturated
    if channels == 1:
        return pixels

    # Separate RGB from Alpha if present
    if channels == 4:
        rgb = pixels[:, :, :3]
        alpha = pixels[:, :, 3:4]
    else:
        rgb = pixels

    # 2. Standard Rec. 709 luminance weights
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    # 3. Create the grayscale baseline
    luminance = np.dot(rgb, weights)[..., np.newaxis]

    # 4. Blend original image toward the grayscale baseline
    desaturated_rgb = (1.0 - factor) * rgb + factor * luminance

    # 5. Reconstruct channels
    if channels == 4:
        return np.concatenate([desaturated_rgb, alpha], axis=2)

    return desaturated_rgb


def calculate_shadow_params(light_vec, max_shadow_distance=10.0):
    lx, ly, lz = light_vec

    # FIX: Use np.maximum instead of the shadowed max() function
    shadow_multiplier = max_shadow_distance / np.maximum(lz, 0.01)

    offset_x = -lx * shadow_multiplier
    offset_y = -ly * shadow_multiplier

    # Blur radius can naturally increase if the shadow gets longer
    blur_radius = 2.0 + (shadow_multiplier * 0.5)

    return (offset_x, offset_y), blur_radius


def array_to_oiio_buf(pixels):
    """
    Ensures a NumPy array has 4 channels (RGBA), sets up the Spec,
    and wraps it cleanly into an OIIO ImageBuf.
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

    # 3. Populate buffer
    buf = oiio.ImageBuf(spec)
    buf.set_pixels(oiio.ROI(0, width, 0, height), fg_pixels)
    return buf, spec, height, width, alpha


def oiio_buf_to_array(buf):
    """Safely reads an OIIO ImageBuf back into a NumPy float32 array."""
    return buf.get_pixels(format=oiio.FLOAT)
