import OpenImageIO as oiio
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, InstanceOf, computed_field
from typing import Optional

NOISE_LIMIT = 0.5


class NoiseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rng: np.random.Generator = Field(...)
    amplitude: float = Field(default=0.5, lt=1.0, gt=0.0)
    texture: Optional[InstanceOf[oiio.ImageBuf]] = None

    @computed_field
    @property
    def norm_amp(self) -> float:
        target_min = 0.0
        target_max = NOISE_LIMIT
        return target_min + (self.amplitude * (target_max - target_min))


def add_noise_to_plane(input_buf, noise_callback, config: NoiseConfig):
    """
    Distorts an OpenImageIO ImageBuf plane using a customizable noise callback.

    Parameters:
    - input_buf (oiio.ImageBuf): The source plane/mask to distort.
    - noise_callback (callable): A function that returns (dx, dy) coordinate offsets.
    - **kwargs: Arbitrary keyword arguments passed directly to the callback (e.g., scale, amplitude).
    """
    spec = input_buf.spec()
    width, height = spec.width, spec.height

    y_indices, x_indices = np.indices((height, width))
    s_coords = x_indices / (width - 1)
    t_coords = y_indices / (height - 1)

    dx_pixels, dy_pixels = noise_callback(
        y_indices=y_indices,
        x_indices=x_indices,
        width=width,
        height=height,
        config=config,
    )

    max_expected_x = float(width)
    max_expected_y = float(height)

    dx_multiplier = config.norm_amp / max_expected_x
    dy_multiplier = config.norm_amp / max_expected_y

    dx = dx_pixels * dx_multiplier
    dy = dy_pixels * dy_multiplier

    st_array = np.stack([s_coords + dx, t_coords + dy], axis=-1)
    st_array = np.ascontiguousarray(st_array, dtype=np.float32)

    st_spec = oiio.ImageSpec(width, height, 2, oiio.FLOAT)
    st_buf = oiio.ImageBuf(st_spec)
    st_buf.set_pixels(oiio.ROI(0, width, 0, height), st_array)

    return oiio.ImageBufAlgo.st_warp(input_buf, st_buf)


def fractal_wave_noise(y_indices, x_indices, width, height, config: NoiseConfig):
    # 1. Generate random phase offsets for each layer using your config's rng.
    # We sample between 0 and 2*pi so the wave can shift anywhere along its cycle.
    phase_x1 = config.rng.uniform(0.0, 2 * np.pi)
    phase_y1 = config.rng.uniform(0.0, 2 * np.pi)

    phase_x2 = config.rng.uniform(0.0, 2 * np.pi)
    phase_y2 = config.rng.uniform(0.0, 2 * np.pi)

    phase_x3 = config.rng.uniform(0.0, 2 * np.pi)
    phase_y3 = config.rng.uniform(0.0, 2 * np.pi)

    # Layer 1: Huge, sweeping clumpy shifts (With Phase Shift)
    dx1 = np.sin(y_indices * 0.015 + phase_x1) * (config.norm_amp * width) * 2.0
    dy1 = np.cos(x_indices * 0.015 + phase_y1) * (config.norm_amp * height) * 2.0

    # Layer 2: Medium jagged details (With Phase Shift)
    dx2 = np.sin(y_indices * 0.1 + phase_x2) * (config.norm_amp * width) * 0.5
    dy2 = np.cos(x_indices * 0.1 + phase_y2) * (config.norm_amp * height) * 0.5

    # Layer 3: High frequency micro-roughness (With Phase Shift)
    dx3 = np.sin(y_indices * 0.4 + phase_x3) * (config.norm_amp * width) * 0.12
    dy3 = np.cos(x_indices * 0.4 + phase_y3) * (config.norm_amp * height) * 0.12

    return (dx1 + dx2 + dx3), (dy1 + dy2 + dy3)


def texture_based_noise(y_indices, x_indices, width, height, config: NoiseConfig):

    height, width = y_indices.shape

    # 1. Automatically resize/crop the texture if its dimensions don't match your target mask
    if config.texture.spec().width != width or config.texture.spec().height != height:
        resized_tex = oiio.ImageBuf()
        oiio.ImageBufAlgo.resize(
            resized_tex, config.texture, roi=oiio.ROI(0, width, 0, height)
        )
        tex_buf_working = resized_tex
    else:
        tex_buf_working = config.texture

    # 2. Extract the pixel values as a 2D NumPy array (Channel 0)
    tex_array = tex_buf_working.get_pixels(oiio.FLOAT)[:, :, 0]

    # 3. Generate Random Offsets for pure variation
    # Pick a random shift for X and Y axes
    shift_y = config.rng.integers(0, height)
    shift_x = config.rng.integers(0, width)

    # 4. Randomize the texture array spatially
    randomized_tex = np.roll(tex_array, shift=(shift_y, shift_x), axis=(0, 1))
    randomized_tex = np.roll(tex_array, shift=(shift_y, shift_x), axis=(1, 0))

    # 5. Create distinct X and Y displacements using the randomized texture
    # Subtracting 0.5 centers the values so the edge pushes both inward and outward
    dx = (randomized_tex - 0.5) * 2.0 * config.norm_amp * width

    # Mirror or invert it for dy so X and Y don't distort identically
    dy = (np.flipud(randomized_tex) - 0.5) * 2.0 * config.norm_amp * height

    return dx, dy
