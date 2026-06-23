import json
import numpy as np
from src.image import CRYPTO_PLANE, COLOR_PLANE
from src.utils import hex_to_float32


# this is final. don't change
def list_cryptopass(image):
    """
    Returns a list of (crypto_id, object_name, hash) tuples found in the EXR file's manifest.
    Example: [("crypto_object", "/box/mesh_0", "d661ca80"), ...]
    """
    # Use the first available plane for metadata
    first_plane = next(iter(image.subimages.values()))
    spec = first_plane["spec"]
    results = []

    for attrib in spec.extra_attribs:
        if attrib.name.startswith("cryptomatte/"):
            parts = attrib.name.split("/")
            if len(parts) < 3:
                continue

            crypto_id = parts[1]
            key_type = parts[2]

            if key_type == "manifest":
                try:
                    manifest_data = json.loads(attrib.value)
                    if isinstance(manifest_data, dict):
                        for name, crypto_hash in manifest_data.items():
                            results.append((crypto_id, name, crypto_hash))
                except (json.JSONDecodeError, TypeError):
                    continue

    return results


# this is final. don't change
def decode_cryptomatte(image, crypto_id, target_hash, plane_name=CRYPTO_PLANE):
    """
    Decodes a specific cryptomatte pass into a mask.
    image: Image object (from src.image)
    crypto_id: The base name of the cryptomatte pass (e.g., "b5b7a6b")
    plane_name: The specific plane to search for the mask (e.g., "CryptoPrimitives00")
    Returns a numpy array (mask) where values range from 0.0 to 1.0 (anti-aliased).
    """
    # Convert hex hash to float using the standard Cryptomatte conversion
    target_float = hex_to_float32(target_hash)

    # Get the specific plane
    plane = image.get_plane(plane_name)
    if not plane:
        first_plane = next(iter(image.subimages.values()))
        return np.zeros(
            (first_plane["pixels"].shape[0], first_plane["pixels"].shape[1]),
            dtype=np.float32,
        )

    pixels = plane["pixels"]

    # Create an empty mask initialized to zeros
    mask = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.float32)

    # Cryptomatte standard channel positions for an isolated multi-part plane:
    # Channel 0 = R (ID 1), Channel 1 = G (Coverage 1)
    # Channel 2 = B (ID 2), Channel 3 = A (Coverage 2)

    # 1. Check Primary ID (Red) and add its Coverage (Green)
    # Using exact matching avoids the 0.0 / tiny float collision bug
    id1_match = pixels[:, :, 0] == target_float
    mask[id1_match] += pixels[:, :, 1][id1_match]

    # 2. Check Secondary ID (Blue) and add its Coverage (Alpha) if there are 4 channels
    if pixels.shape[2] >= 4:
        id2_match = pixels[:, :, 2] == target_float
        mask[id2_match] += pixels[:, :, 3][id2_match]

    # Clip the mask to 0.0 - 1.0 range just to be safe from floating-point errors
    return np.clip(mask, 0.0, 1.0)


def mask_crypto_pass(image, mask, plane=COLOR_PLANE):
    """
    Masks the image with red color based on the provided mask.
    image: Image object (from src.image)
    mask: numpy array (height, width)
    plane: The plane name to apply the mask to.
    """
    plane_data = image.get_plane(plane)
    if not plane_data:
        return None

    # Create a copy of the pixels to modify
    pixels = plane_data["pixels"].copy()

    # Create a boolean mask for indexing
    mask_bool = mask == 0.0

    # Apply red color where mask is > 0
    # We only modify RGB channels (0, 1, 2)
    if pixels.shape[2] >= 3:
        pixels[mask_bool, 0] = 1.0  # Red
        pixels[mask_bool, 1] = 0.0  # Green
        pixels[mask_bool, 2] = 0.0  # Blue

    return pixels


def apply_masked_operation(pixels, mask, op_callback):
    """
    Applies an operation callback to pixels where the mask is > 0.
    pixels: numpy array (height, width, channels)
    mask: numpy array (height, width)
    op_callback: function(pixels) -> pixels
    """
    # Create a copy to avoid modifying original data
    result_pixels = pixels.copy()

    # Create a boolean mask for indexing
    mask_bool = mask > 0.0

    # Extract masked pixels
    masked_region = result_pixels[mask_bool]

    # Apply operation
    modified_region = op_callback(masked_region)

    # Put back into result
    result_pixels[mask_bool] = modified_region

    return result_pixels
