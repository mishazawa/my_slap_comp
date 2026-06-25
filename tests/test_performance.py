import pytest
import time
from src.core import slap_comp
from src.image import Image
from src.settings import (
    SHADOW_COLOR,
    SHADOW_INTENSITY,
    OUTLINE_THICKNESS,
    OUTLINE_COLOR,
    MASK_SMOOTH_WIDTH,
    MASK_SMOOTH_HEIGHT,
    EXECUTOR_SEQUENCE,
    EXECUTOR_THREAD,
)
from src.globals import init_global_textures


@pytest.fixture(autouse=True)
def setup_globals():
    # Initialize with dummy paths for testing purposes
    init_global_textures("./test_data/noise", "./test_data/paper")


@pytest.fixture
def mock_image():
    # Assuming a test image exists at this path, or replace with a valid path
    try:
        return Image.read("./test_data/0001.exr")
    except Exception:
        return None


def test_sequential_performance(mock_image):
    if mock_image is None:
        pytest.skip("No image provided")

    start = time.perf_counter()
    slap_comp(
        mock_image,
        shadow_color=SHADOW_COLOR,
        shadow_intensity=SHADOW_INTENSITY,
        outline_thickness=OUTLINE_THICKNESS,
        outline_color=OUTLINE_COLOR,
        mask_smooth_width=MASK_SMOOTH_WIDTH,
        mask_smooth_height=MASK_SMOOTH_HEIGHT,
        executor_type=EXECUTOR_SEQUENCE,
    )
    print(f"Sequential time: {time.perf_counter() - start}")


def test_thread_performance(mock_image):
    if mock_image is None:
        pytest.skip("No image provided")

    start = time.perf_counter()
    slap_comp(
        mock_image,
        shadow_color=SHADOW_COLOR,
        shadow_intensity=SHADOW_INTENSITY,
        outline_thickness=OUTLINE_THICKNESS,
        outline_color=OUTLINE_COLOR,
        mask_smooth_width=MASK_SMOOTH_WIDTH,
        mask_smooth_height=MASK_SMOOTH_HEIGHT,
        executor_type=EXECUTOR_THREAD,
    )
    print(f"Thread time: {time.perf_counter() - start}")
