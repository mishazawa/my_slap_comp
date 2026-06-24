import OpenImageIO as oiio
import numpy as np

from src.noise import (
    add_noise_to_plane,
    fractal_wave_noise,
    texture_based_noise,
)
from settings import (
    FRACTAL_WAVE_AMPLITUDE_REL,
    TEXTURE_BASED_AMPLITUDE_REL,
    LIGHT,
    PAPER_SCALE,
)
from src.utils import (
    desaturate_pixels,
    calculate_shadow_params,
    array_to_oiio_buf,
    oiio_buf_to_array,
)


LIGHT_VECTOR = np.array(LIGHT, dtype=np.float32)

# Ensure it's a unit vector (length of 1) for clean math
LIGHT_VECTOR /= np.linalg.norm(LIGHT_VECTOR)


def smooth_mask(mask, width=5, height=5):
    """
    Smooths a binary or grayscale mask using a median filter.
    Expects a 2D numpy array (height, width).
    """
    height_px, width_px = mask.shape

    spec = oiio.ImageSpec(width_px, height_px, 1, oiio.FLOAT)
    mask_buf = oiio.ImageBuf(spec)

    roi = oiio.ROI(0, width_px, 0, height_px)
    mask_buf.set_pixels(roi, mask.astype(np.float32).flatten())

    median_mask = oiio.ImageBuf()
    oiio.ImageBufAlgo.median_filter(median_mask, mask_buf, width=width, height=height)

    return median_mask.get_pixels(oiio.FLOAT).reshape(height_px, width_px)


def add_shadow_to_a_layer(
    pixels,
    shadow_color=(0.0, 0.0, 0.0),
    shadow_intensity=0.5,
):
    """Adds a soft customizable shadow to a layer using OpenImageIO ImageBufAlgo.

    Parameters:
    - pixels (np.ndarray): Input image array (RGB or RGBA).
    - offset (tuple): (y_offset, x_offset) in pixels to shift the shadow.
    - blur_radius (float): Softness of the shadow.
    - shadow_color (tuple): (R, G, B) normalized color of the shadow (e.g., (0,0,0) for black).
    - shadow_intensity (float): Max opacity of the shadow (0.0 to 1.0).
    """
    from src.globals import textures

    if textures is None:
        raise RuntimeError("Global textures were never initialized!")

    fg_buf, spec, height, width, alpha = array_to_oiio_buf(pixels)

    offset, blur_radius = calculate_shadow_params(LIGHT_VECTOR)
    y_off, x_off = offset
    y_off = int(np.round(y_off))
    x_off = int(np.round(x_off))
    shifted_alpha = np.roll(alpha, shift=(y_off, x_off), axis=(0, 1))

    if y_off > 0:
        shifted_alpha[:y_off, :] = 0
    elif y_off < 0:
        shifted_alpha[y_off:, :] = 0

    if x_off > 0:
        shifted_alpha[:, :x_off] = 0
    elif x_off < 0:
        shifted_alpha[:, x_off:] = 0

    rgba_mask_pixels = np.concatenate([shifted_alpha] * 4, axis=2)

    shadow_mask_buf = oiio.ImageBuf(spec)
    shadow_mask_buf.set_pixels(oiio.ROI(0, width, 0, height), rgba_mask_pixels)

    shadow_mask_buf = add_noise_to_plane(
        shadow_mask_buf,
        noise_callback=fractal_wave_noise,
        amplitude=FRACTAL_WAVE_AMPLITUDE_REL,
    )

    if blur_radius > 0:
        K = oiio.ImageBufAlgo.make_kernel("gaussian", blur_radius, blur_radius)
        blurred_shadow_mask = oiio.ImageBuf(spec)
        oiio.ImageBufAlgo.convolve(blurred_shadow_mask, shadow_mask_buf, K)
    else:
        blurred_shadow_mask = shadow_mask_buf

    blurred_shadow_mask = add_noise_to_plane(
        blurred_shadow_mask,
        noise_callback=texture_based_noise,
        texture_buf=textures.noise(),
        amplitude=TEXTURE_BASED_AMPLITUDE_REL,
    )

    shadow_color_buf = oiio.ImageBuf(spec)

    fill_color = (
        shadow_color[0],
        shadow_color[1],
        shadow_color[2],
        shadow_intensity,
    )
    oiio.ImageBufAlgo.fill(shadow_color_buf, fill_color)

    shadow_final = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.mul(shadow_final, shadow_color_buf, blurred_shadow_mask)

    final_composite = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.over(final_composite, shadow_final, final_composite)
    oiio.ImageBufAlgo.over(final_composite, fg_buf, final_composite)

    return oiio_buf_to_array(final_composite)


def add_outline_to_layer(
    pixels,
    outline_thickness=5,
    outline_color=(1.0, 1.0, 1.0),
):
    """
    Adds an outline using a clean mask-first approach:
    1. Extract a solid binary mask from the layer's alpha.
    2. Dilate the mask using OpenImageIO.
    3. Build a solid color backing based on the dilated mask.
    4. Composite the original pixels over it.
    """
    from src.globals import textures

    if textures is None:
        raise RuntimeError("Global textures were never initialized!")
    orig_buf, spec, height, width, alpha = array_to_oiio_buf(pixels)

    mask_buf = oiio.ImageBuf(spec)
    mask_buf.set_pixels(
        oiio.ROI(0, width, 0, height), np.concatenate([alpha] * 4, axis=2)
    )

    if outline_thickness > 0:
        eroded_mask = oiio.ImageBufAlgo.erode(mask_buf, width=outline_thickness)

        eroded_fg_buf = oiio.ImageBuf(spec)
        oiio.ImageBufAlgo.mul(eroded_fg_buf, orig_buf, eroded_mask)
        fg_to_composite = eroded_fg_buf

        base_for_dilation = eroded_mask
    else:
        fg_to_composite = orig_buf
        base_for_dilation = mask_buf

    K = oiio.ImageBufAlgo.make_kernel("gaussian", 5, 5)
    dilated_mask = oiio.ImageBufAlgo.dilate(base_for_dilation, width=outline_thickness)
    blurred_shadow_mask = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.convolve(blurred_shadow_mask, dilated_mask, K)

    blurred_shadow_mask = add_noise_to_plane(
        blurred_shadow_mask,
        noise_callback=texture_based_noise,
        texture_buf=textures.noise(),
        amplitude=TEXTURE_BASED_AMPLITUDE_REL,
    )

    outline_buf = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.fill(outline_buf, (*outline_color, 1.0))
    oiio.ImageBufAlgo.mul(outline_buf, outline_buf, blurred_shadow_mask)

    final_composite = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.over(final_composite, fg_to_composite, outline_buf)

    return oiio_buf_to_array(final_composite)


def apply_paper(pixels, strength=1):
    from src.globals import textures

    if textures is None:
        raise RuntimeError("Global textures were never initialized!")
    pixels = desaturate_pixels(pixels)
    orig_buf, spec, height, width, alpha = array_to_oiio_buf(pixels)

    tex_buf = textures.paper()
    texture_raw = tex_buf.get_pixels(format=oiio.FLOAT)
    texture_raw = desaturate_pixels(texture_raw, 1)
    tex_h, tex_w = texture_raw.shape[0], texture_raw.shape[1]

    angle = np.random.uniform(0.0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    y_idx, x_idx = np.indices((height, width))
    norm_x = (x_idx / (width - 1)) - 0.5
    norm_y = (y_idx / (height - 1)) - 0.5

    scale = PAPER_SCALE
    rot_x = norm_x * cos_a - norm_y * sin_a
    rot_y = norm_x * sin_a + norm_y * cos_a

    tex_x_coords = np.mod(((rot_x * scale + 0.5) * (tex_w - 1)).astype(np.int32), tex_w)
    tex_y_coords = np.mod(((rot_y * scale + 0.5) * (tex_h - 1)).astype(np.int32), tex_h)

    bg_paper = texture_raw[tex_y_coords, tex_x_coords]

    if bg_paper.ndim == 2:
        bg_paper = np.stack([bg_paper] * 3, axis=-1)
    else:
        bg_paper = bg_paper[:, :, :3]

    fg_rgb = pixels[:, :, :3]
    paper_look = fg_rgb * bg_paper
    blended_rgb = (strength * paper_look) + ((1.0 - strength) * fg_rgb)
    final_rgb = blended_rgb * alpha + bg_paper * (1.0 - alpha)
    final_rgba = np.concatenate([final_rgb, alpha], axis=2)

    final_composite = oiio.ImageBuf(spec)
    final_composite.set_pixels(oiio.ROI(0, width, 0, height), final_rgba)
    return oiio_buf_to_array(final_composite)
