from pathlib import Path
import OpenImageIO as oiio


class Textures:
    def __init__(self, noise_dir, paper_dir):
        """
        Initializes the texture pool by scanning directory folders.

        :param noise_dir: Path string or Path object to the folder containing noise maps.
        :param paper_dir: Path string or Path object to the folder containing paper textures.
        """
        # Supported image extensions (add more if your assets use different formats)
        valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr"}

        # 1. Scan and lazy-load noise textures
        self._noise_pool = [
            oiio.ImageBuf(str(p))
            for p in Path(noise_dir).iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        # 2. Scan and lazy-load paper textures
        self._paper_pool = [
            oiio.ImageBuf(str(p))
            for p in Path(paper_dir).iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        # Simple safety checks to warn you if a folder was empty
        if not self._noise_pool:
            print(f"Warning: No valid textures found in noise directory: {noise_dir}")
        if not self._paper_pool:
            print(f"Warning: No valid textures found in paper directory: {paper_dir}")

    def noise(self, random):
        """Returns a random noise ImageBuf from the pool."""
        if not self._noise_pool:
            return None
        return random.choice(self._noise_pool)

    def paper(self, random):
        """Returns a random paper ImageBuf from the pool."""
        if not self._paper_pool:
            return None
        return random.choice(self._paper_pool)
