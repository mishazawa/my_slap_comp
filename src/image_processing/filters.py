import OpenImageIO as oiio
import numpy as np

from pydantic import BaseModel, Field, ConfigDict
from typing import Tuple, Optional, List

from src.settings import (
    NOISE_SCALE,
    SEED,
    SHADOW_COLOR,
    SHADOW_INTENSITY,
    OUTLINE_THICKNESS,
    OUTLINE_COLOR,
    MASK_SMOOTH_WIDTH,
    MASK_SMOOTH_HEIGHT,
    LIGHT,
    PAPER_SCALE,
    PAPER_STRENGTH,
    SHADOW_LIMIT,
)

from .noise import (
    add_noise_to_plane,
    fractal_wave_noise,
    NoiseConfig,
    texture_based_noise,
)


class FilterConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int = Field(default=SEED, description="Seed for reproducibility")

    # 1. Declare rng as a Field that defaults to None, allowing you to pass it in later
    rng: Optional[np.random.Generator] = Field(default=None, exclude=True)

    shadow_color: Tuple[float, float, float] = Field(default=SHADOW_COLOR)
    shadow_intensity: float = Field(default=SHADOW_INTENSITY, ge=0.0, le=1.0)
    outline_thickness: int = Field(default=OUTLINE_THICKNESS, ge=0)
    outline_color: Tuple[float, float, float] = Field(default=OUTLINE_COLOR)
    mask_smooth_width: int = Field(default=MASK_SMOOTH_WIDTH, ge=1)
    mask_smooth_height: int = Field(default=MASK_SMOOTH_HEIGHT, ge=1)
    light_vector: List[float] = Field(default=LIGHT)
    paper_strength: float = Field(default=PAPER_STRENGTH, ge=0.0, le=1.0)
    paper_scale: float = Field(default=PAPER_SCALE, ge=0.0)
    noise_scale: float = Field(default=NOISE_SCALE, ge=0.0, le=1.0)
    shadow_limit: float = Field(default=SHADOW_LIMIT, ge=0.0)


def desaturate_pixels(pixels, factor=0.15):
    """
    Slightly desaturates an RGB or RGBA image. Safe for single-channel inputs.
    """
    # 1. Check channels
    channels = pixels.shape[2] if pixels.ndim == 3 else 1

    # If the image is already single-channel (grayscale), it cannot be desaturated
    if channels == 1:
        return pixels

    # Separate RGB from Alpha if present
    if channels == 4:
        rgb = pixels[:, :, :3]
        alpha = pixels[:, :, 3:4]
    else:
        rgb = pixels

    # 2. Standard Rec. 709 luminance weights
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    # 3. Create the grayscale baseline
    luminance = np.dot(rgb, weights)[..., np.newaxis]

    # 4. Blend original image toward the grayscale baseline
    desaturated_rgb = (1.0 - factor) * rgb + factor * luminance

    # 5. Reconstruct channels
    if channels == 4:
        return np.concatenate([desaturated_rgb, alpha], axis=2)

    return desaturated_rgb


def smooth_mask(mask_buf: oiio.ImageBuf, config: FilterConfig) -> oiio.ImageBuf:
    """
    Accepts an ImageBuf, runs the filter entirely in C++,
    and returns a brand new ImageBuf. Zero RAM copying back to Python!
    """
    median_mask = oiio.ImageBuf()
    oiio.ImageBufAlgo.median_filter(
        median_mask,
        mask_buf,
        width=config.mask_smooth_width,
        height=config.mask_smooth_height,
    )
    return median_mask


def add_shadow(color_buf: oiio.ImageBuf, config: FilterConfig) -> oiio.ImageBuf:
    """Adds a soft customizable shadow to an image layer entirely inside OpenImageIO.
    Utilizes 3x3 matrix warping for sub-pixel precision spatial translation.
    Parameters:
    - color_buf (oiio.ImageBuf): Input image buffer (RGBA).
    - shadow_color (tuple): (R, G, B) normalized color of the shadow.
    - shadow_intensity (float): Max opacity of the shadow (0.0 to 1.0).
    Returns:
    - oiio.ImageBuf: The final alpha-composited layer with the shadow underneath.
    """
    from src.globals import textures

    if textures is None:
        raise RuntimeError("Global textures were never initialized!")

    spec = color_buf.spec()

    y_off, x_off, penumbra = calculate_shadow_params(config)

    mask_buf = oiio.ImageBuf()
    oiio.ImageBufAlgo.channels(mask_buf, color_buf, channelorder=(3, 3, 3, 3))

    M = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, x_off, y_off, 1.0)

    shadow_mask_buf = oiio.ImageBufAlgo.warp(mask_buf, M, filtername="", wrap="default")

    shadow_mask_buf = add_noise_to_plane(
        shadow_mask_buf,
        noise_callback=fractal_wave_noise,
        config=NoiseConfig(
            amplitude=config.noise_scale,
            rng=config.rng,
        ),
    )

    if penumbra > 0:
        K = oiio.ImageBufAlgo.make_kernel("gaussian", penumbra, penumbra)
        blurred_shadow_mask = oiio.ImageBuf()
        oiio.ImageBufAlgo.convolve(blurred_shadow_mask, shadow_mask_buf, K)
    else:
        blurred_shadow_mask = shadow_mask_buf

    blurred_shadow_mask = add_noise_to_plane(
        blurred_shadow_mask,
        noise_callback=texture_based_noise,
        config=NoiseConfig(
            rng=config.rng,
            texture=textures.noise(config.rng),
            amplitude=config.noise_scale,
        ),
    )

    shadow_color_buf = oiio.ImageBuf()
    fill_color = config.shadow_color + (config.shadow_intensity,)
    oiio.ImageBufAlgo.fill(shadow_color_buf, fill_color, roi=spec.roi)

    shadow_final = oiio.ImageBuf()
    oiio.ImageBufAlgo.mul(shadow_final, shadow_color_buf, blurred_shadow_mask)

    final_composite = oiio.ImageBuf()
    oiio.ImageBufAlgo.over(final_composite, color_buf, shadow_final)
    return final_composite


def add_outline(color_buf: oiio.ImageBuf, config: FilterConfig) -> oiio.ImageBuf:
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

    spec = color_buf.spec()

    mask_buf = oiio.ImageBuf()
    oiio.ImageBufAlgo.channels(mask_buf, color_buf, channelorder=(3, 3, 3, 3))

    if config.outline_thickness > 0:
        eroded_mask = oiio.ImageBufAlgo.erode(mask_buf, width=config.outline_thickness)

        eroded_fg_buf = oiio.ImageBuf()
        oiio.ImageBufAlgo.mul(eroded_fg_buf, color_buf, eroded_mask)
        fg_to_composite = eroded_fg_buf

        base_for_dilation = eroded_mask
    else:
        fg_to_composite = color_buf
        base_for_dilation = mask_buf

    dilated_mask = oiio.ImageBufAlgo.dilate(
        base_for_dilation, width=config.outline_thickness
    )

    K = oiio.ImageBufAlgo.make_kernel("gaussian", 5, 5)
    blurred_shadow_mask = oiio.ImageBuf()
    oiio.ImageBufAlgo.convolve(blurred_shadow_mask, dilated_mask, K)

    blurred_shadow_mask = add_noise_to_plane(
        blurred_shadow_mask,
        noise_callback=texture_based_noise,
        config=NoiseConfig(
            rng=config.rng,
            texture=textures.noise(config.rng),
            amplitude=config.noise_scale,
        ),
    )

    outline_buf = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.fill(outline_buf, (*config.outline_color, 1.0), roi=spec.roi)
    oiio.ImageBufAlgo.mul(outline_buf, outline_buf, blurred_shadow_mask)

    final_composite = oiio.ImageBuf()
    oiio.ImageBufAlgo.over(final_composite, fg_to_composite, outline_buf)

    return final_composite


def apply_paper(color_buf: oiio.ImageBuf, config: FilterConfig) -> oiio.ImageBuf:
    from src.globals import textures

    if textures is None:
        raise RuntimeError("Global textures were never initialized!")

    pixels = color_buf.get_pixels(format=oiio.FLOAT)
    height, width, _ = pixels.shape

    pixels = desaturate_pixels(pixels)

    tex_buf = textures.paper(config.rng)
    texture_raw = tex_buf.get_pixels(format=oiio.FLOAT)
    texture_raw = desaturate_pixels(texture_raw, 1)
    tex_h, tex_w = texture_raw.shape[0], texture_raw.shape[1]

    angle = config.rng.uniform(0.0, 2 * np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    y_idx, x_idx = np.indices((height, width))
    norm_x = (x_idx / (width - 1)) - 0.5
    norm_y = (y_idx / (height - 1)) - 0.5

    scale = config.paper_scale
    rot_x = norm_x * cos_a - norm_y * sin_a
    rot_y = norm_x * sin_a + norm_y * cos_a

    tex_x_coords = np.mod(((rot_x * scale + 0.5) * (tex_w - 1)).astype(np.int32), tex_w)
    tex_y_coords = np.mod(((rot_y * scale + 0.5) * (tex_h - 1)).astype(np.int32), tex_h)

    bg_paper = texture_raw[tex_y_coords, tex_x_coords]

    if bg_paper.ndim == 2:
        bg_paper = np.stack([bg_paper] * 3, axis=-1)
    else:
        bg_paper = bg_paper[:, :, :3]

    alpha = pixels[:, :, 3:4]
    fg_rgb = pixels[:, :, :3]

    paper_look = fg_rgb * bg_paper
    blended_rgb = (config.paper_strength * paper_look) + (
        (1.0 - config.paper_strength) * fg_rgb
    )
    final_rgb = blended_rgb * alpha + bg_paper * (1.0 - alpha)

    final_rgba = np.concatenate([final_rgb, alpha], axis=2)

    output_buf = oiio.ImageBuf(final_rgba.astype(np.float32, copy=False))
    return output_buf


def calculate_shadow_params(config: FilterConfig):
    lv = np.array(config.light_vector, dtype=np.float32)
    norm = np.linalg.norm(lv)

    if norm > 0:
        lv /= norm
    else:
        lv = np.array([0.0, 0.0, 1.0])

    lx, ly, lz = lv

    shadow_multiplier = config.shadow_limit / np.maximum(lz, 0.01)

    offset_x = float(-lx * shadow_multiplier)
    offset_y = float(-ly * shadow_multiplier)

    actual_distance = np.sqrt(offset_x**2 + offset_y**2)

    if actual_distance > config.shadow_limit:
        scale_factor = config.shadow_limit / actual_distance
        offset_x *= scale_factor
        offset_y *= scale_factor
        shadow_multiplier = scale_factor

    blur_radius = 2.0 + (shadow_multiplier * 0.5)

    return offset_x, offset_y, blur_radius
