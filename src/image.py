import OpenImageIO as oiio

COLOR_PLANE = "directemission"
CRYPTO_PLANE = "CryptoPrimitives00"
DEPTH_PLANE = "depth"
UV_PLANE = "st"


class Image:
    def __init__(self, subimages):
        """
        subimages: A dictionary where keys are plane names and values are dictionaries:
            'pixels': numpy array
            'channel_names': list of strings
            'spec': oiio.ImageSpec
        """
        self._subimages = subimages

    @classmethod
    def read(cls, filepath):
        inp = oiio.ImageInput.open(filepath)
        if not inp:
            raise FileNotFoundError(f"Could not open file: {filepath}")

        subimages = {}
        subimage_idx = 0

        while inp.seek_subimage(subimage_idx, 0):
            spec = inp.spec()

            name = spec.get_string_attribute("oiio:subimagename")
            if not name:
                name = f"subimage_{subimage_idx}"

            pixels = inp.read_image(oiio.FLOAT)
            if pixels is None:
                break

            subimages[name] = {
                "pixels": pixels,
                "channel_names": spec.channelnames,
                "spec": spec,
            }

            subimage_idx += 1

        inp.close()

        return cls(subimages)

    @property
    def subimages(self):
        return self._subimages

    def get_plane(self, name):
        return self._subimages.get(name)
