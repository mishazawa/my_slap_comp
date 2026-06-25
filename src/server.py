import os
import OpenImageIO as oiio
from fastapi import FastAPI, UploadFile, Response
from contextlib import asynccontextmanager
import tempfile
import shutil

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

    return write_image_to_buffer(final_img, format_extension="png")


@app.post("/process/")
async def process_image(filepath: str):  # ai! rewrite endpoint accept file path
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".exr") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Process using the file path
        image_bytes = process_image_file(tmp_path)
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return Response(content=image_bytes, media_type="image/png")
