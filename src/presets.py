from src.image_processing.filters import (
    add_outline,
    add_shadow,
    apply_paper,
    smooth_mask,
    FilterConfig,
)
from src.image_processing.masks import create_mask_buf, get_masked_pixels
from src.utils import ensure_rgba_buf


# just grab masked pixels
def noop_preset(task_tuple, img, color_plane, config: FilterConfig):
    _, _, name, target_hash = task_tuple

    mask = create_mask_buf(img, target_hash)
    a = ensure_rgba_buf(color_plane["pixels"])
    a = get_masked_pixels(a, mask)
    return a


def cutout_element_preset(task_tuple, img, color_plane, config: FilterConfig):
    _, _, name, target_hash = task_tuple

    mask = create_mask_buf(img, target_hash)
    a = ensure_rgba_buf(color_plane["pixels"])
    a = get_masked_pixels(a, mask)
    a = get_masked_pixels(apply_paper(a, config), mask)
    a = add_outline(a, config)
    a = add_shadow(a, config)

    return a
