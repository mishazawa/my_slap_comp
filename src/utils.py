import struct
import os
import OpenImageIO as oiio
import numpy as np
from src.image import Image, COLOR_PLANE


def map_hip_to_working_dir(filepath: str) -> str:
    working_dir = os.getenv("WORKING_DIR", "/app/working_dir")
    if filepath.startswith("$HIP"):
        return filepath.replace("$HIP", working_dir)
    return filepath


def map_working_dir_to_pdg(filepath: str) -> str:
    working_dir = os.getenv("WORKING_DIR", "/app/working_dir")
    if filepath.startswith(working_dir):
        return filepath.replace(working_dir, "$PDG_DIR", 1)
    return filepath


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
