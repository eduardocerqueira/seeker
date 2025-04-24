#date: 2025-04-24T16:48:29Z
#url: https://api.github.com/gists/91c97de01016af8e09ae96c4ae894756
#owner: https://api.github.com/users/abuibrahimjega

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
import cv2
import easyocr
import numpy as np
import os
import uuid
import shutil
from typing import List, Optional
import pyttsx3  # Add import for pyttsx3

app = FastAPI(
    title="Image Processing API",
    description="API for image processing including text removal and text extraction",
    version="1.0.0"
)

# Configure folders
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
AUDIO_FOLDER = 'audio'  # Add audio folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)  # Create audio folder

# Initialize EasyOCR Reader (English only by default)
reader = None
current_langs = []

def get_reader(languages=['en']):
    global reader, current_langs
    
    # Convert to sorted tuple for consistent comparison
    langs_tuple = tuple(sorted(languages))
    
    # Check if we need to initialize a new reader
    if reader is None or set(current_langs) != set(languages):
        print(f"Initializing EasyOCR reader with languages: {languages}")
        reader = easyocr.Reader(languages, gpu=False)
        current_langs = languages
    
    return reader

def remove_text_from_image(image_path, languages=['en'], inpaint_radius=3):
    """
    Remove text from an image using EasyOCR and inpainting
    
    Args:
        image_path: Path to the input image
        languages: List of language codes for OCR
        inpaint_radius: Radius for inpainting algorithm
    
    Returns:
        Path to the processed image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Get or initialize EasyOCR Reader
    reader = get_reader(languages)
    
    # Run OCR
    results = reader.readtext(img)
    
    # Create mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # Loop through all detected texts
    for (bbox, text, prob) in results:
        # Unpack the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)  # Fill detected text area on mask
    
    # Inpaint to remove text
    inpainted = cv2.inpaint(img, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    
    # Generate output path
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(RESULT_FOLDER, f"{name}_cleaned{ext}")
    
    # Save the result
    cv2.imwrite(output_path, inpainted)
    
    return output_path

@app.post("/remove-text", summary="Remove text from an image")
async def remove_text_api(
    image: UploadFile = File(...),
    languages: str = Form("en"),
    inpaint_radius: int = Form(3)
):
    """
    Remove text from an uploaded image.
    
    - **image**: The image file to process
    - **languages**: Comma-separated list of language codes (default: 'en')
    - **inpaint_radius**: Radius for inpainting algorithm (default: 3)
    
    Returns the processed image with text removed.
    """
    try:
        # Parse language list
        language_list = languages.split(',')
        
        # Generate a unique filename and save the uploaded file
        filename = image.filename
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Process the image
        try:
            result_path = remove_text_from_image(file_path, language_list, inpaint_radius)
            
            # Return the processed image
            return FileResponse(
                result_path,
                media_type="image/jpeg",
                filename=os.path.basename(result_path)
            )
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-text", summary="Extract text from an image")
async def extract_text_api(
    image: UploadFile = File(...),
    languages: str = Form("en")
):
    """
    Extract text from an uploaded image.
    
    - **image**: The image file to process
    - **languages**: Comma-separated list of language codes (default: 'en')
    
    Returns the extracted text.
    """
    try:
        # Parse language list
        language_list = languages.split(',')
        
        # Generate a unique filename and save the uploaded file
        filename = image.filename
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Process the image
        try:
            # Get the reader
            reader = get_reader(language_list)
            
            # Read the image
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError(f"Could not read image from {file_path}")
            
            # Run OCR
            results = reader.readtext(img)
            
            # Extract text
            extracted_text = [text for _, text, _ in results]
            
            # Return the extracted text
            return {"text": extracted_text, "full_text": " ".join(extracted_text)}
        
        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add text-to-speech functions
def init_engine():
    engine = pyttsx3.init()
    return engine

def text_to_speech(text, rate=150, volume=1.0):
    """
    Convert text to speech using pyttsx3
    
    Args:
        text: Text to convert to speech
        rate: Speech rate (default: 150)
        volume: Speech volume (default: 1.0)
    
    Returns:
        Path to the audio file
    """
    engine = init_engine()
    
    # Set properties
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    
    # Create a temporary file for the audio
    audio_path = os.path.join(AUDIO_FOLDER, f"{uuid.uuid4()}.mp3")
    
    # Generate speech
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    
    return audio_path

@app.post("/speak", summary="Convert text to speech")
async def speak_api(
    text: str = Form(...),
    rate: int = Form(150, description="Speech rate"),
    volume: float = Form(1.0, description="Speech volume (0-1)")
):
    """
    Convert text to speech using pyttsx3 and return audio.
    
    - **text**: The text to convert to speech
    - **rate**: Speech rate (default: 150)
    - **volume**: Speech volume (0-1, default: 1.0)
    
    Returns the audio file.
    """
    try:
        # Convert text to speech
        audio_path = text_to_speech(text, rate, volume)
        
        # Return the audio file
        return FileResponse(
            audio_path,
            media_type="audio/mp3",
            filename=os.path.basename(audio_path)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices", summary="Get available voices")
async def get_voices():
    """
    Get a list of available voices for pyttsx3.
    
    Returns a list of voice IDs and names.
    """
    try:
        engine = init_engine()
        voices = engine.getProperty('voices')
        
        voice_list = []
        for voice in voices:
            voice_list.append({
                "id": voice.id,
                "name": voice.name
            })
        
        return {"voices": voice_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Check if the API is running.
    
    Returns a status message.
    """
    return {"status": "ok", "services": ["text_removal", "text_extraction", "text_to_speech"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 