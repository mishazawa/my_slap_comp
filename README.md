# Compositing Pipeline

A Python script for processing EXR files using OpenImageIO. It extracts Cryptomatte layers, applies effects, and composites them based on depth.

## Features

- **Cryptomatte**: Decodes passes from EXR metadata.
- **Depth Sorting**: Orders layers by average depth for correct occlusion.
- **Effects**:
  - Paper textures
  - Outlines
  - Drop shadows
- **I/O**: Uses OpenImageIO for reading/writing.

## Requirements

- [OpenImageIO](https://openimageio.readthedocs.io/)
- NumPy
- uv

## Usage

### Direct Execution
```bash
uv run main.py <input_path.exr> [--noise-dir <dir>] [--paper-dir <dir>]
```

### Makefile Commands
- `make run`: Runs the pipeline on `./test_data/0001.exr`.
- `make seq`: Processes all EXR files in `./test_data/sequence`.

## Pipeline Steps

1. **Extraction**: Decodes Cryptomatte metadata.
2. **Sorting**: Calculates average depth per layer to determine stacking order.
3. **Processing**: Applies effects (noise, paper, outlines, shadows) to each layer.
4. **Compositing**: Merges layers using OIIO's `over` operation.
5. **Output**: Converts to sRGB and saves as PNG.
