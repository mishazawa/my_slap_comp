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
- **API**: Includes a FastAPI server for remote processing.

## Requirements

- [OpenImageIO](https://openimageio.readthedocs.io/)
- NumPy
- uv

## Usage

### Running the Server
The project runs as a FastAPI server. Start it using:
```bash
uv run main.py
```

### API
The server exposes a `/process/` endpoint.
- **Endpoint**: `POST /process/`
- **Payload**:
  ```json
  {
    "filepath": "./test_data/0001.exr"
  }
  ```
- Use the provided `api.http` file with the REST Client extension in VS Code or similar tools to test.

### Makefile Commands
- `make run`: Starts the API server.
- `make test`: Runs the test suite.
- `make pdg`: Starts the docker-compose environment.

## Pipeline Steps

1. **Extraction**: Decodes Cryptomatte metadata.
2. **Sorting**: Calculates average depth per layer to determine stacking order.
3. **Processing**: Applies effects (noise, paper, outlines, shadows) to each layer.
4. **Compositing**: Merges layers using OIIO's `over` operation.
5. **Output**: Converts to sRGB and saves as PNG.
