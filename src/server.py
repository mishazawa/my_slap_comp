import os
import tempfile
import OpenImageIO as oiio
from fastapi import FastAPI, UploadFile, Response
from contextlib import asynccontextmanager

from src.globals import init_global_textures
from src.core import slap_comp
from src.image import Image, COLOR_PLANE
from src.utils import write_image

# Configuration for textures, defaulting to standard paths if not set
NOISE_DIR = os.getenv("NOISE_DIR", "assets/noise")
PAPER_DIR = os.getenv("PAPER_DIR", "assets/paper")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Allocate resources at startup
    init_global_textures(NOISE_DIR, PAPER_DIR)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/process/")
async def process_image(file: UploadFile):
    # Read uploaded file content
    content = await file.read()
    
    # Use a temporary file for processing as the pipeline expects a file path
    with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as tmp_in:
        tmp_in.write(content)
        input_path = tmp_in.name

    try:
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
        
        # Write to a temporary file to capture the output bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_out:
            write_image(tmp_out.name, final_img)
            output_path = tmp_out.name
            
        with open(output_path, "rb") as f:
            image_bytes = f.read()
            
        # Clean up output temp file
        os.remove(output_path)
        
        return Response(content=image_bytes, media_type="image/png")
        
    finally:
        # Clean up input temp file
        if os.path.exists(input_path):
            os.remove(input_path)
