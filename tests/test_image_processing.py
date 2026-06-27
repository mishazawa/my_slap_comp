import pytest
from pathlib import Path
from src.utils import read_image, write_image, oiio_buf_to_image, median
from src.image_processing.crypto import decode_cryptomatte, list_cryptopass
from src.image_processing.filters import FilterConfig
from src.core import slap_comp
from src.globals import init_global_textures
from src.presets import cutout_element_preset
import numpy as np
import src.settings as settings

MOCK_IMAGE = "./test_data/solids.exr"


@pytest.fixture(autouse=True)
def setup_globals():
    # Initialize with dummy paths for testing purposes
    init_global_textures("./test_data/test", "./test_data/test")


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
    mask = decode_cryptomatte(mock_image, target_hash)

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
        sort_func=median,
        pass_processor=cutout_element_preset,
        config=FilterConfig(
            seed=settings.SEED,
            rng=np.random.default_rng(settings.SEED),
            light_vector=[0, 0, 0],
            shadow_color=(0, 0, 1),
            shadow_intensity=1,
            outline_thickness=5,
            outline_color=(1, 0, 0),
            mask_smooth_width=1,
            mask_smooth_height=1,
            noise_scale=0.1,
        ),
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
