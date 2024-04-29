#date: 2024-04-29T16:52:29Z
#url: https://api.github.com/gists/b4f311572e12fa76f68dbaf094b6799c
#owner: https://api.github.com/users/remmy36

import os
import logging
import configparser
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import openai

# Initialize the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    raise ValueError("OpenAI API key is not set in environment variables.")
class DetailedLogFilter(logging.Filter):
    """Custom filter to ignore detailed metadata and stream logs."""
    def filter(self, record):
        unwanted_log_parts = ['IHDR', 'IDAT', 'pHYs', 'iTXt', 'tEXt', 'tag:', 'STREAM']
        return not any(part in record.getMessage() for part in unwanted_log_parts)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.DEBUG, filename='/Users/rem/Workspace/process_log.log',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.addFilter(DetailedLogFilter())
    logging.info("Logging system configured.")

def load_config():
    """Load configuration settings from a file."""
    config = configparser.ConfigParser()
    if not config.read('/Users/rem/Workspace/config.ini'):
        logging.error("Failed to load config file.")
        raise ValueError("Failed to load config file.")
    return config

def load_model():
    """Load the MobileNetV2 model for image classification."""
    model = MobileNetV2(weights='imagenet')
    logging.info("MobileNetV2 model loaded.")
    return model

def classify_image(model, img_path):
    """Classify an image using the MobileNetV2 model and handle potential errors."""
    try:
        logging.debug(f"Classifying image: {img_path}")
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)
        top_prediction = decoded_predictions[0][0][1] if decoded_predictions else 'uncategorized'
        logging.debug(f"Top prediction: {top_prediction}")
        return top_prediction
    except Exception as e:
        logging.error(f"An error occurred while classifying the image {img_path}: {str(e)}")
        return None

def generate_description(prompt):
    """Generate a descriptive text for an image using OpenAI's GPT model."""
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens= "**********"
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Failed to generate description: {str(e)}")
        return "No description available."

def process_image(model, config, filepath):
    """Process and classify images from a specified directory."""
    try:
        logging.debug(f"Processing image: {filepath}")
        img = Image.open(filepath)
        img.verify()  # This checks for the integrity of the image
        img.close()

        category = classify_image(model, filepath)
        description = generate_description(f"Describe this image: {filepath} in detail.")
        if category:
            target_dir = os.path.join(config['Settings']['BaseStorageDir'], category)
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, os.path.basename(filepath))
            img = Image.open(filepath)
            img.save(new_path)
            img.close()
            logging.info(f"Image moved: {new_path}")
            logging.info(f"Generated description: {description}")
            os.remove(filepath)  # Remove the original file after moving
    except Exception as e:
        logging.error(f"Error processing image {filepath}: {e}")

def sweep_directory(model, config):
    """Scan a directory and process all images."""
    dir_path = config['Settings']['DownloadPath']
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            filepath = os.path.join(root, name)
            process_image(model, config, filepath)
        for name in dirs:
            dirpath = os.path.join(root, name)
            if not os.listdir(dirpath):  # Check if directory is empty
                os.rmdir(dirpath)  # Remove the directory if it's empty
                logging.info(f"Removed empty directory: {dirpath}")

if __name__ == "__main__":
    setup_logging()
    config = load_config()
    model = load_model()
    sweep_directory(model, config)
