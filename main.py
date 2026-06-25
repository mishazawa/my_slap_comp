import os
import uvicorn
from src.globals import init_global_textures
from src.server import app

# Initialize global textures from environment variables
noise_dir = os.environ.get("NOISE_DIR")
paper_dir = os.environ.get("PAPER_DIR")

init_global_textures(noise_dir, paper_dir)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
