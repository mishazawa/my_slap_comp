import pytest
from pathlib import Path
from src.utils import read_image, write_image, oiio_buf_to_image
from src.image_processing.filters import FilterConfig
from src.core import slap_comp
from src.globals import init_global_textures
from src.presets import cutout_element_preset
import numpy as np
import src.settings as settings

MOCK_IMAGE = "./test_data/test_sorting.exr"


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


def test_slap_comp_execution(mock_image, tmp_path):
    if mock_image is None:
        pytest.skip("No image provided")

    # Run slap_comp
    final_gamma_buffer = slap_comp(
        mock_image,
        pass_processor=cutout_element_preset,
        config=FilterConfig(
            seed=settings.SEED,
            rng=np.random.default_rng(settings.SEED),
            shadow_color=(0, 0, 0),
            shadow_intensity=0.8,
            outline_thickness=2,
            outline_color=(0.9, 0.7, 0.7),
            noise_scale=0.05,
            paper_strength=0.5,
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
