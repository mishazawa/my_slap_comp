import struct
import OpenImageIO as oiio
import numpy as np
from src.image import Image, COLOR_PLANE
from src.settings import SHADOW_LIMIT


def read_image(filepath):
    """
    Reads an EXR image into an Image object.
    """
    return Image.read(filepath)


def write_image(filepath, image):
    """
    Writes an Image object to a file using its embedded OIIO spec.
    """
    plane_name = next(iter(image.subimages))
    plane = image.get_plane(plane_name)

    output_file = oiio.ImageOutput.create(filepath)
    if not output_file:
        raise RuntimeError(f"Could not create output file: {oiio.geterror()}")

    # Use the spec directly from your plane dictionary
    spec = plane["spec"]

    output_file.open(filepath, spec)
    try:
        output_file.write_image(plane["pixels"])
    finally:
        output_file.close()


def oiio_buf_to_image(buf, plane_name=COLOR_PLANE):
    """
    Converts an OIIO ImageBuf to an Image object.
    """
    pixels = buf.get_pixels(oiio.FLOAT)
    spec = buf.spec()
    return Image(
        {
            plane_name: {
                "pixels": pixels,
                "channel_names": ["R", "G", "B", "A"],
                "spec": spec,
            }
        }
    )


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


def median(data):
    return np.median(data)


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


def get_masked_pixels(
    color_buf: oiio.ImageBuf, mask_buf: oiio.ImageBuf
) -> oiio.ImageBuf:
    """
    Returns an RGBA ImageBuf where the masked region is extracted from color_buf.
    Both inputs are expected to be OpenImageIO ImageBuf objects.
    """
    # 1. Multiply color pixels by mask opacity
    masked_color = oiio.ImageBuf()
    oiio.ImageBufAlgo.mul(masked_color, color_buf, mask_buf)

    # 2. Isolate just the R, G, B channels from your multiplied result
    # Python syntax expects an explicit tuple of channels to copy or reorder
    rgb_only = oiio.ImageBufAlgo.channels(masked_color, (0, 1, 2))

    # 3. Force your grayscale mask_buf into a single channel buffer
    # Just in case mask_buf was initialized with 4 channels elsewhere
    alpha_only = oiio.ImageBufAlgo.channels(mask_buf, (0,))

    # 4. Append the alpha channel onto the back of the RGB channels
    # This automatically builds a perfect 4-channel RGBA output image
    result = oiio.ImageBuf()
    oiio.ImageBufAlgo.channel_append(result, rgb_only, alpha_only)

    return result


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


def calculate_shadow_params(light_vec, max_shadow_distance=SHADOW_LIMIT):
    lx, ly, lz = light_vec

    # FIX: Use np.maximum instead of the shadowed max() function
    shadow_multiplier = max_shadow_distance / np.maximum(lz, 0.01)

    offset_x = -lx * shadow_multiplier
    offset_y = -ly * shadow_multiplier

    # Blur radius can naturally increase if the shadow gets longer
    blur_radius = 2.0 + (shadow_multiplier * 0.5)

    return (offset_x, offset_y), blur_radius


def ensure_rgba_buf(pixels):
    """
    Converts a NumPy array to RGBA if needed, and WRAPS it into an ImageBuf.
    This creates an OIIO buffer that views the same memory space where possible.
    """
    height, width, channels = pixels.shape

    if channels == 3:
        # We must allocate here because we are adding a channel
        alpha = np.ones((height, width, 1), dtype=np.float32)
        fg_pixels = np.concatenate([pixels, alpha], axis=2)
    else:
        fg_pixels = pixels.astype(np.float32, copy=False)

    # Directly initializing ImageBuf with a NumPy array bypasses `set_pixels` copies!
    buf = oiio.ImageBuf(fg_pixels)

    # Force channel names onto the spec if needed
    buf.specmod().channelnames = ["R", "G", "B", "A"]

    return buf
