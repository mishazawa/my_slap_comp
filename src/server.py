import os
import io
import OpenImageIO as oiio
from fastapi import FastAPI, UploadFile, Response
from contextlib import asynccontextmanager

from src.globals import init_global_textures
from src.core import slap_comp
from src.image import Image, COLOR_PLANE
from src.utils import write_image_to_buffer

# Configuration for textures, defaulting to standard paths if not set
NOISE_DIR = os.getenv("NOISE_DIR")
PAPER_DIR = os.getenv("PAPER_DIR")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Allocate resources at startup
    init_global_textures(NOISE_DIR, PAPER_DIR)
    yield


app = FastAPI(lifespan=lifespan)


def process_image_file(input_buffer):
    # Run the pipeline logic
    img = Image.read(input_buffer)
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

    return write_image_to_buffer(final_img, format_extension="png")


@app.post("/process/")
async def process_image(file: UploadFile):
    # Read uploaded file content
    content = await file.read()
    input_buffer = io.BytesIO(content)

    # Process in memory
    image_bytes = process_image_file(input_buffer)

    return Response(content=image_bytes, media_type="image/png")
