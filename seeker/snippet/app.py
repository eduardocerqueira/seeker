#date: 2023-06-27T17:06:07Z
#url: https://api.github.com/gists/c3a0cbb43db49fc85c98075c5f380514
#owner: https://api.github.com/users/trohit

#!/usr/bin/env python3
from fastapi import File, UploadFile, Request, FastAPI
from fastapi.templating import Jinja2Templates
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    base64_encoded_image = base64.b64encode(contents).decode("utf-8")

    return templates.TemplateResponse("display.html", {"request": request,  "myImage": base64_encoded_image})

