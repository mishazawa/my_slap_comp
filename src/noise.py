import OpenImageIO as oiio
import numpy as np


def add_noise_to_plane(input_buf, noise_callback, **kwargs):
    """
    Distorts an OpenImageIO ImageBuf plane using a customizable noise callback.

    Parameters:
    - input_buf (oiio.ImageBuf): The source plane/mask to distort.
    - noise_callback (callable): A function that returns (dx, dy) coordinate offsets.
    - **kwargs: Arbitrary keyword arguments passed directly to the callback (e.g., scale, amplitude).
    """
    spec = input_buf.spec()
    width, height = spec.width, spec.height

    # 1. Generate normalized S and T base grids (0.0 to 1.0)
    y_indices, x_indices = np.indices((height, width))
    s_coords = x_indices / (width - 1)
    t_coords = y_indices / (height - 1)

    # 2. Execute the callback to get non-uniform pixel offsets
    dx_pixels, dy_pixels = noise_callback(
        y_indices, x_indices, width=width, height=height, **kwargs
    )

    # 3. Normalize pixel displacements into 0-1 ST coordinate space
    dx = dx_pixels / (width - 1)
    dy = dy_pixels / (height - 1)

    # 4. Stack into an ST texture coordinate array
    st_array = np.stack([s_coords + dx, t_coords + dy], axis=-1).astype(np.float32)

    # 5. Pack coordinates into an OIIO ImageBuf and warp
    st_spec = oiio.ImageSpec(width, height, 2, oiio.FLOAT)
    st_buf = oiio.ImageBuf(st_spec)
    st_buf.set_pixels(oiio.ROI(0, width, 0, height), st_array)

    return oiio.ImageBufAlgo.st_warp(input_buf, st_buf)


def fractal_wave_noise(y_indices, x_indices, amplitude=8.0, width=1, height=1):
    # Layer 1: Huge, sweeping clumpy shifts
    dx1 = np.sin(y_indices * 0.015) * (amplitude * width) * 2.0
    dy1 = np.cos(x_indices * 0.015) * (amplitude * height) * 2.0

    # Layer 2: Medium jagged details
    dx2 = np.sin(y_indices * 0.1) * (amplitude * width) * 0.5
    dy2 = np.cos(x_indices * 0.1) * (amplitude * height) * 0.5

    # Layer 3: High frequency micro-roughness
    dx3 = np.sin(y_indices * 0.4) * (amplitude * width) * 0.12
    dy3 = np.cos(x_indices * 0.4) * (amplitude * height) * 0.12

    return (dx1 + dx2 + dx3), (dy1 + dy2 + dy3)


def sheared_noise(y_indices, x_indices, amplitude=10.0, width=1, height=1):
    # Modulate frequency based on both axes simultaneously to break uniformity
    dx = np.sin((y_indices * 0.05) + (x_indices * 0.01)) * (amplitude * width)
    dy = np.cos((x_indices * 0.04) - (y_indices * 0.02)) * ((amplitude * height) * 0.5)
    return dx, dy


def texture_based_noise(
    y_indices, x_indices, texture_buf, amplitude=15.0, width=1, height=1
):
    """
    Uses an existing OpenImageIO ImageBuf texture to drive non-uniform mask imperfections
    with a randomized spatial offset and rotation.

    Parameters:
    - texture_buf (oiio.ImageBuf): The pre-loaded image buffer containing noise/grunge.
    - amplitude (float): Maximum pixel displacement depth.
    """
    height, width = y_indices.shape

    # 1. Automatically resize/crop the texture if its dimensions don't match your target mask
    if texture_buf.spec().width != width or texture_buf.spec().height != height:
        resized_tex = oiio.ImageBuf()
        oiio.ImageBufAlgo.resize(
            resized_tex, texture_buf, roi=oiio.ROI(0, width, 0, height)
        )
        tex_buf_working = resized_tex
    else:
        tex_buf_working = texture_buf

    # 2. Extract the pixel values as a 2D NumPy array (Channel 0)
    tex_array = tex_buf_working.get_pixels(oiio.FLOAT)[:, :, 0]

    # 3. Generate Random Offsets for pure variation
    # Pick a random shift for X and Y axes
    shift_y = np.random.randint(0, height)
    shift_x = np.random.randint(0, width)

    # 4. Randomize the texture array spatially
    randomized_tex = np.roll(tex_array, shift=(shift_y, shift_x), axis=(0, 1))
    randomized_tex = np.roll(tex_array, shift=(shift_y, shift_x), axis=(1, 0))

    # 5. Create distinct X and Y displacements using the randomized texture
    # Subtracting 0.5 centers the values so the edge pushes both inward and outward
    dx = (randomized_tex - 0.5) * 2.0 * amplitude * width

    # Mirror or invert it for dy so X and Y don't distort identically
    dy = (np.flipud(randomized_tex) - 0.5) * 2.0 * amplitude * height

    return dx, dy
