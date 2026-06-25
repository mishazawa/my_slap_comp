import os
import OpenImageIO as oiio
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.globals import init_global_textures
from src.core import slap_comp
from src.image import Image, COLOR_PLANE
from src.utils import write_image

# Configuration for textures, defaulting to standard paths if not set
NOISE_DIR = os.getenv("NOISE_DIR")
PAPER_DIR = os.getenv("PAPER_DIR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Allocate resources at startup
    init_global_textures(NOISE_DIR, PAPER_DIR)
    yield


app = FastAPI(lifespan=lifespan)


def process_image_file(input_path):
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


# ai: i want this endpoint to grab input data from request body
# for now i want to configure pipeline using input parameters
# described in settings.py
# i want them to be default to constants from settings.py ai!
@app.post("/process/")
async def process_image(filepath: str):
    output_path = process_image_file(filepath)
    return {"output_path": output_path}
