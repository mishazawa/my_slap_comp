import sys
import os
import argparse

import OpenImageIO as oiio
from src.globals import init_global_textures
from src.core import slap_comp
from src.image import Image, COLOR_PLANE
from src.utils import write_image

# ai: rewrite this file to use of server.py ai!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the compositing pipeline on an EXR image."
    )

    parser.add_argument("input_path", type=str, help="Path to the input .exr file")

    parser.add_argument(
        "-n",
        "--noise-dir",
        type=str,
        help="Directory containing noise textures",
    )
    parser.add_argument(
        "-p",
        "--paper-dir",
        type=str,
        help="Directory containing paper textures",
    )

    args = parser.parse_args()

    init_global_textures(args.noise_dir, args.paper_dir)

    input_path = sys.argv[1]

    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}.png"

    img = Image.read(input_path)
    final_gamma_buffer = slap_comp(img)

    write_image(
        output_path,
        Image(
            {
                COLOR_PLANE: {
                    "pixels": final_gamma_buffer.get_pixels(oiio.FLOAT),
                    "channel_names": ["R", "G", "B", "A"],
                }
            }
        ),
    )
    print(f"Saved masked image to {output_path}")
