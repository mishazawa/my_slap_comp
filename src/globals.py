textures = None


def init_global_textures(noise_dir, paper_dir):
    """Initializes the global texture storage once at startup."""
    from src.textures import Textures

    global textures
    textures = Textures(noise_dir, paper_dir)
