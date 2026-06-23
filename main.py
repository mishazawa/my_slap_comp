import sys
import os
import OpenImageIO as oiio
from src.core import slap_comp
from src.image import Image, COLOR_PLANE
from src.utils import write_image

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_exr_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Generate output path: replace extension with .png
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
