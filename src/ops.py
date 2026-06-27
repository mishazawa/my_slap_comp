import OpenImageIO as oiio
import numpy as np

from src.noise import (
    add_noise_to_plane,
    fractal_wave_noise,
    texture_based_noise,
)
from src.settings import (
    FRACTAL_WAVE_AMPLITUDE_REL,
    TEXTURE_BASED_AMPLITUDE_REL,
    LIGHT,
    PAPER_SCALE,
)
from src.utils import (
    calculate_shadow_params,
)


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


def smooth_mask(mask_buf: oiio.ImageBuf, width=5, height=5) -> oiio.ImageBuf:
    """
    Accepts an ImageBuf, runs the filter entirely in C++,
    and returns a brand new ImageBuf. Zero RAM copying back to Python!
    """
    median_mask = oiio.ImageBuf()
    oiio.ImageBufAlgo.median_filter(median_mask, mask_buf, width=width, height=height)
    return median_mask


def add_shadow(
    color_buf: oiio.ImageBuf,
    shadow_color=(0.0, 0.0, 0.0),
    shadow_intensity=0.5,
    light_vector=LIGHT,
) -> oiio.ImageBuf:
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

    # Ensure it's a unit vector (length of 1) for clean math
    lv = np.array(light_vector, dtype=np.float32)
    lv /= np.linalg.norm(lv)

    offset, blur_radius = calculate_shadow_params(lv)
    y_off, x_off = offset

    mask_buf = oiio.ImageBuf()
    oiio.ImageBufAlgo.channels(mask_buf, color_buf, channelorder=(3, 3, 3, 3))

    M = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, x_off, y_off, 1.0)

    shadow_mask_buf = oiio.ImageBufAlgo.warp(mask_buf, M, filtername="", wrap="default")

    shadow_mask_buf = add_noise_to_plane(
        shadow_mask_buf,
        noise_callback=fractal_wave_noise,
        amplitude=FRACTAL_WAVE_AMPLITUDE_REL,
    )

    if blur_radius > 0:
        K = oiio.ImageBufAlgo.make_kernel("gaussian", blur_radius, blur_radius)
        blurred_shadow_mask = oiio.ImageBuf()
        oiio.ImageBufAlgo.convolve(blurred_shadow_mask, shadow_mask_buf, K)
    else:
        blurred_shadow_mask = shadow_mask_buf

    blurred_shadow_mask = add_noise_to_plane(
        blurred_shadow_mask,
        noise_callback=texture_based_noise,
        texture_buf=textures.noise(),
        amplitude=TEXTURE_BASED_AMPLITUDE_REL,
    )

    shadow_color_buf = oiio.ImageBuf()
    fill_color = shadow_color + (shadow_intensity,)
    oiio.ImageBufAlgo.fill(shadow_color_buf, fill_color, roi=spec.roi)

    shadow_final = oiio.ImageBuf()
    oiio.ImageBufAlgo.mul(shadow_final, shadow_color_buf, blurred_shadow_mask)

    final_composite = oiio.ImageBuf()
    oiio.ImageBufAlgo.over(final_composite, color_buf, shadow_final)
    return final_composite


def add_outline(
    color_buf: oiio.ImageBuf,
    outline_thickness=5,
    outline_color=(1.0, 1.0, 1.0),
) -> oiio.ImageBuf:
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

    if outline_thickness > 0:
        eroded_mask = oiio.ImageBufAlgo.erode(mask_buf, width=outline_thickness)

        eroded_fg_buf = oiio.ImageBuf()
        oiio.ImageBufAlgo.mul(eroded_fg_buf, color_buf, eroded_mask)
        fg_to_composite = eroded_fg_buf

        base_for_dilation = eroded_mask
    else:
        fg_to_composite = color_buf
        base_for_dilation = mask_buf

    dilated_mask = oiio.ImageBufAlgo.dilate(base_for_dilation, width=outline_thickness)

    K = oiio.ImageBufAlgo.make_kernel("gaussian", 5, 5)
    blurred_shadow_mask = oiio.ImageBuf()
    oiio.ImageBufAlgo.convolve(blurred_shadow_mask, dilated_mask, K)

    blurred_shadow_mask = add_noise_to_plane(
        blurred_shadow_mask,
        noise_callback=texture_based_noise,
        texture_buf=textures.noise(),
        amplitude=TEXTURE_BASED_AMPLITUDE_REL,
    )

    outline_buf = oiio.ImageBuf(spec)
    oiio.ImageBufAlgo.fill(outline_buf, (*outline_color, 1.0), roi=spec.roi)
    oiio.ImageBufAlgo.mul(outline_buf, outline_buf, blurred_shadow_mask)

    final_composite = oiio.ImageBuf()
    oiio.ImageBufAlgo.over(final_composite, fg_to_composite, outline_buf)

    return final_composite


def apply_paper(color_buf: oiio.ImageBuf, strength=1.0) -> oiio.ImageBuf:
    from src.globals import textures

    if textures is None:
        raise RuntimeError("Global textures were never initialized!")

    pixels = color_buf.get_pixels(format=oiio.FLOAT)
    height, width, _ = pixels.shape

    pixels = desaturate_pixels(pixels)

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

    alpha = pixels[:, :, 3:4]
    fg_rgb = pixels[:, :, :3]

    paper_look = fg_rgb * bg_paper
    blended_rgb = (strength * paper_look) + ((1.0 - strength) * fg_rgb)
    final_rgb = blended_rgb * alpha + bg_paper * (1.0 - alpha)

    final_rgba = np.concatenate([final_rgb, alpha], axis=2)

    output_buf = oiio.ImageBuf(final_rgba.astype(np.float32, copy=False))
    return output_buf
