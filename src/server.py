# ai:
# i want to wrap this pipeline into web server
# i want to use fast api
# lets bootstrap server code for single endpoint
# i want here the code from @main.py used as endpoint callback
# i need to allocate resources (globals.py textures) at the moment server starts
# no file writes allowed. only response with file bytes (currently png) ai!
from typing import Annotated

from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}
