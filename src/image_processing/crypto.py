import json
import numpy as np
from src.utils import hex_to_float32
from .constants import CRYPTO_PLANE


def decode_cryptomatte(image, target_hash, plane_name=CRYPTO_PLANE):
    """
    Decodes a specific cryptomatte pass into a mask.
    image: Image object (from src.image)
    plane_name: The specific plane to search for the mask (e.g., "CryptoPrimitives00")
    Returns a numpy array (mask) where values range from 0.0 to 1.0 (anti-aliased).
    """
    # Convert hex hash to float using the standard Cryptomatte conversion
    target_float = hex_to_float32(target_hash)

    matching_planes = [
        name for name in image.subimages.keys() if name.startswith(plane_name)
    ]

    if not matching_planes:
        first_plane = next(iter(image.subimages.values()))
        return np.zeros(
            (first_plane["pixels"].shape[0], first_plane["pixels"].shape[1]),
            dtype=np.float32,
        )

    first_plane_pixels = image.get_plane(matching_planes[0])["pixels"]
    height, width, _ = first_plane_pixels.shape
    mask = np.zeros((height, width), dtype=np.float32)
    for name in matching_planes:
        plane = image.get_plane(name)
        if not plane or "pixels" not in plane:
            continue

        pixels = plane["pixels"]
        _, _, channels = pixels.shape

        # Pair 1: ID is channel 0, Coverage is channel 1
        if channels >= 2:
            id1_idx = np.where(pixels[:, :, 0] == target_float)
            mask[id1_idx] += pixels[:, :, 1][id1_idx]

        # Pair 2: ID is channel 2, Coverage is channel 3
        if channels >= 4:
            id2_idx = np.where(pixels[:, :, 2] == target_float)
            mask[id2_idx] += pixels[:, :, 3][id2_idx]

    # Ensure anti-aliased edges don't accidentally sum past 1.0
    np.clip(mask, 0.0, 1.0, out=mask)
    return mask

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