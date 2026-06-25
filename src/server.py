import os
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Tuple

from src.globals import init_global_textures
from src.core import slap_comp
from src.utils import write_image, read_image, oiio_buf_to_image
from src.settings import (
    LIGHT,
    SHADOW_COLOR,
    SHADOW_INTENSITY,
    SHADOW_LIMIT,
    OUTLINE_THICKNESS,
    OUTLINE_COLOR,
    MASK_SMOOTH_WIDTH,
    MASK_SMOOTH_HEIGHT,
    FRACTAL_WAVE_AMPLITUDE_REL,
    TEXTURE_BASED_AMPLITUDE_REL,
    PAPER_SCALE,
)

# Configuration for textures, defaulting to standard paths if not set
NOISE_DIR = os.getenv("NOISE_DIR")
PAPER_DIR = os.getenv("PAPER_DIR")


class ProcessRequest(BaseModel):
    filepath: str
    shadow_color: Optional[Tuple[float, float, float]] = SHADOW_COLOR
    shadow_intensity: Optional[float] = SHADOW_INTENSITY
    shadow_limit: Optional[int] = SHADOW_LIMIT
    outline_thickness: Optional[int] = OUTLINE_THICKNESS
    outline_color: Optional[Tuple[float, float, float]] = OUTLINE_COLOR
    mask_smooth_width: Optional[int] = MASK_SMOOTH_WIDTH
    mask_smooth_height: Optional[int] = MASK_SMOOTH_HEIGHT
    fractal_wave_amplitude_rel: Optional[float] = FRACTAL_WAVE_AMPLITUDE_REL
    texture_based_amplitude_rel: Optional[float] = TEXTURE_BASED_AMPLITUDE_REL
    paper_scale: Optional[float] = PAPER_SCALE
    light_vector: Optional[List[float]] = LIGHT


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Allocate resources at startup
    init_global_textures(NOISE_DIR, PAPER_DIR)
    yield


app = FastAPI(lifespan=lifespan)


def process_image_file(input_path, **kwargs):

    img = read_image(input_path)
    final_gamma_buffer = slap_comp(img, **kwargs)

    output_path = str(Path(input_path).with_suffix(".png"))
    write_image(
        output_path,
        oiio_buf_to_image(final_gamma_buffer),
    )

    return output_path


@app.post("/process/")
async def process_image(request: ProcessRequest):
    # Process using the file path and parameters
    params = request.dict()
    filepath = params.pop("filepath")
    output_path = process_image_file(filepath, **params)
    return {"output_path": output_path}
