import os
import OpenImageIO as oiio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Tuple

from src.globals import init_global_textures
from src.core import slap_comp
from src.image import Image, COLOR_PLANE
from src.utils import write_image
import settings

# Configuration for textures, defaulting to standard paths if not set
NOISE_DIR = os.getenv("NOISE_DIR")
PAPER_DIR = os.getenv("PAPER_DIR")


class ProcessRequest(BaseModel):
    filepath: str
    light: Optional[List[float]] = settings.LIGHT
    shadow_color: Optional[Tuple[float, float, float]] = settings.SHADOW_COLOR
    shadow_intensity: Optional[float] = settings.SHADOW_INTENSITY
    shadow_limit: Optional[int] = settings.SHADOW_LIMIT
    outline_thickness: Optional[int] = settings.OUTLINE_THICKNESS
    outline_color: Optional[Tuple[float, float, float]] = settings.OUTLINE_COLOR
    mask_smooth_width: Optional[int] = settings.MASK_SMOOTH_WIDTH
    mask_smooth_height: Optional[int] = settings.MASK_SMOOTH_HEIGHT
    fractal_wave_amplitude_rel: Optional[float] = settings.FRACTAL_WAVE_AMPLITUDE_REL
    texture_based_amplitude_rel: Optional[float] = settings.TEXTURE_BASED_AMPLITUDE_REL
    paper_scale: Optional[float] = settings.PAPER_SCALE


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Allocate resources at startup
    init_global_textures(NOISE_DIR, PAPER_DIR)
    yield


app = FastAPI(lifespan=lifespan)


def process_image_file(input_path, **kwargs):
    # Run the pipeline logic
    img = Image.read(input_path)
    final_gamma_buffer = slap_comp(img)

    # Prepare the output image
    final_img = Image(
        {
            COLOR_PLANE: {
                "pixels": final_gamma_buffer.get_pixels(oiio.FLOAT),
                "channel_names": ["R", "G", "B", "A"],
            }
        }
    )

    # 4) server save file to the same location with other extension
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}.png"
    write_image(output_path, final_img)

    # 5) server respond with created file filepath
    return output_path


@app.post("/process/")
async def process_image(request: ProcessRequest):
    # Process using the file path and parameters
    params = request.dict()
    filepath = params.pop("filepath")
    output_path = process_image_file(filepath, **params)
    return {"output_path": output_path}
