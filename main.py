import os
import argparse

from src.globals import init_global_textures
from server import process_image_file

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

    input_path = args.input_path

    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}.png"

    process_image_file(input_path, output_path)
    print(f"Saved masked image to {output_path}")
