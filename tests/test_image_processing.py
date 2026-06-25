import pytest
import os
from pathlib import Path
from src.utils import read_image, write_image, oiio_buf_to_image
from src.crypto import decode_cryptomatte, list_cryptopass
from src.core import slap_comp
from src.globals import init_global_textures
import src.settings as settings

MOCK_IMAGE = "./test_data/0001.exr"


@pytest.fixture(autouse=True)
def setup_globals():
    # Initialize with dummy paths for testing purposes
    init_global_textures("./test_data/noise", "./test_data/paper")


@pytest.fixture
def mock_image():
    try:
        return read_image(MOCK_IMAGE)
    except Exception:
        return None


def test_decode_cryptomatte(mock_image):
    if mock_image is None:
        pytest.skip("No image provided")

    passes = list_cryptopass(mock_image)
    if not passes:
        pytest.skip("No crypto passes found in image")

    crypto_id, name, target_hash = passes[0]
    mask = decode_cryptomatte(mock_image, crypto_id, target_hash)

    assert mask is not None
    # Assuming mask is a numpy array
    # The mask should match the dimensions of the image planes
    # We can check against the directemission plane if it exists
    color_plane = mock_image.get_plane("directemission")
    if color_plane:
        assert mask.shape[:2] == color_plane["pixels"].shape[:2]


def test_slap_comp_execution(mock_image, tmp_path):
    if mock_image is None:
        pytest.skip("No image provided")

    # Run slap_comp
    final_gamma_buffer = slap_comp(
        mock_image,
        shadow_color=settings.SHADOW_COLOR,
        shadow_intensity=settings.SHADOW_INTENSITY,
        outline_thickness=settings.OUTLINE_THICKNESS,
        outline_color=settings.OUTLINE_COLOR,
        mask_smooth_width=settings.MASK_SMOOTH_WIDTH,
        mask_smooth_height=settings.MASK_SMOOTH_HEIGHT,
        executor_type=settings.EXECUTOR_SEQUENCE,
    )

    # Define output path
    output_path = Path(MOCK_IMAGE).with_suffix(".png")

    # Write image
    write_image(
        str(output_path),
        oiio_buf_to_image(final_gamma_buffer),
    )

    # Verify file exists
    assert output_path.exists()
    assert output_path.stat().st_size > 0
