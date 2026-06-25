import json
import numpy as np
from src.image import CRYPTO_PLANE
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

    height, width, channels = pixels.shape
    mask = np.zeros((height, width), dtype=np.float32)

    id1_idx = np.where(pixels[:, :, 0] == target_float)
    mask[id1_idx] += pixels[:, :, 1][id1_idx]

    if channels >= 4:
        id2_idx = np.where(pixels[:, :, 2] == target_float)
        mask[id2_idx] += pixels[:, :, 3][id2_idx]

    np.clip(mask, 0.0, 1.0, out=mask)

    return mask
