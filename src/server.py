import os
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

from pydantic import Field

from src.globals import init_global_textures
from src.core import slap_comp
from src.utils import (
    write_image,
    read_image,
    oiio_buf_to_image,
    map_hip_to_working_dir,
    map_working_dir_to_pdg,
    median,
    average,
)
from src.presets import cutout_element_preset, noop_preset


from src.image_processing.filters import FilterConfig

NOISE_DIR = os.getenv("NOISE_DIR")
PAPER_DIR = os.getenv("PAPER_DIR")


class ProcessRequest(FilterConfig):
    filepath: str = Field(..., description="Path to the source image file")

    def to_filter_config(self) -> FilterConfig:
        """Converts the incoming request parameters to a backend FilterConfig."""
        request_data = self.model_dump()
        request_data.pop("filepath", None)
        request_data["rng"] = np.random.default_rng(request_data["seed"])
        return FilterConfig(**request_data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_global_textures(NOISE_DIR, PAPER_DIR)
    yield


app = FastAPI(lifespan=lifespan)


def process_image_file(input_path, sort_func, pass_processor, config: FilterConfig):

    img = read_image(input_path)
    final_gamma_buffer = slap_comp(
        img,
        sort_func=sort_func,
        pass_processor=pass_processor,
        config=config,
    )
    output_path = str(Path(input_path).with_suffix(".png"))
    write_image(
        output_path,
        oiio_buf_to_image(final_gamma_buffer),
    )

    return output_path


def process_input(request: ProcessRequest, sort_func, pass_processor):
    params = request.dict()
    filepath = params.pop("filepath")
    filepath = map_hip_to_working_dir(filepath)
    output_path = process_image_file(
        filepath,
        sort_func=sort_func,
        pass_processor=pass_processor,
        config=request.to_filter_config(),
    )
    output_path = map_working_dir_to_pdg(output_path)
    return {"output_path": output_path}


@app.post("/process/")
async def process_image(request: ProcessRequest):
    return process_input(request, median, cutout_element_preset)


@app.post("/process/background")
async def process_bg(request: ProcessRequest):
    return process_input(request, average, cutout_element_preset)


@app.post("/process/transparend")
async def process_transparent(request: ProcessRequest):
    return process_input(request, median, noop_preset)
