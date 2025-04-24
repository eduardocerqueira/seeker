#date: 2025-04-24T16:48:29Z
#url: https://api.github.com/gists/91c97de01016af8e09ae96c4ae894756
#owner: https://api.github.com/users/abuibrahimjega

import requests
import argparse
import os
from PIL import Image
import io

def remove_text_from_image(image_path, api_url, languages='en', inpaint_radius=3):
    """
    Send an image to the API to remove text
    
    Args:
        image_path: Path to the input image
        api_url: Base URL of the API
        languages: Comma-separated list of language codes (e.g., 'en,fr')
        inpaint_radius: Radius for inpainting algorithm
    
    Returns:
        Path to the processed image
    """
    endpoint = f"{api_url}/remove-text"
    
    # Prepare the form data
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'languages': languages,
            'inpaint_radius': str(inpaint_radius)
        }
        
        # Send the request
        response = requests.post(endpoint, files=files, data=data)
    
    # Check if request was successful
    if response.status_code == 200:
        # Get the output filename
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"{name}_cleaned{ext}"
        
        # Save the response content as an image
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Text removed successfully! Saved as {output_path}")
        return output_path
    else:
        try:
            error_data = response.json()
            print(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
        except:
            print(f"❌ Error: {response.text}")
        return None

def extract_text_from_image(image_path, api_url, languages='en'):
    """
    Send an image to the API to extract text
    
    Args:
        image_path: Path to the input image
        api_url: Base URL of the API
        languages: Comma-separated list of language codes (e.g., 'en,fr')
    
    Returns:
        Dictionary with the extracted text
    """
    endpoint = f"{api_url}/extract-text"
    
    # Prepare the form data
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'languages': languages
        }
        
        # Send the request
        response = requests.post(endpoint, files=files, data=data)
    
    # Check if request was successful
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Text extracted successfully!")
        print(f"Full text: {result['full_text']}")
        return result
    else:
        try:
            error_data = response.json()
            print(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
        except:
            print(f"❌ Error: {response.text}")
        return None

def text_to_speech(text, api_url, engine='gtts', language='en', rate=200, slow=False):
    """
    Send text to the API to convert to speech
    
    Args:
        text: Text to convert to speech
        api_url: Base URL of the API
        engine: TTS engine to use ('gtts' or 'pyttsx3')
        language: Language code for gTTS or voice ID for pyttsx3
        rate: Speech rate (pyttsx3 only)
        slow: Speak slowly (gTTS only)
    
    Returns:
        Path to the audio file
    """
    endpoint = f"{api_url}/speak"
    
    # Prepare the form data
    data = {
        'text': text,
        'engine': engine,
        'language': language,
        'rate': str(rate),
        'slow': str(slow).lower()
    }
    
    # Send the request
    response = requests.post(endpoint, data=data)
    
    # Check if request was successful
    if response.status_code == 200:
        # Save the audio file
        audio_path = f"speech_{engine}_{language}.mp3"
        with open(audio_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Speech generated successfully! Saved as {audio_path}")
        return audio_path
    else:
        try:
            error_data = response.json()
            print(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
        except:
            print(f"❌ Error: {response.text}")
        return None

def process_image_to_speech(image_path, api_url, ocr_langs='en', tts_engine='gtts', tts_lang='en'):
    """
    Complete workflow: Extract text from image and convert it to speech
    
    Args:
        image_path: Path to the input image
        api_url: Base URL of the API
        ocr_langs: Language codes for OCR
        tts_engine: TTS engine to use
        tts_lang: Language for TTS
    
    Returns:
        Path to the audio file
    """
    # Extract text from image
    result = extract_text_from_image(image_path, api_url, ocr_langs)
    
    if result and result.get('full_text'):
        # Convert text to speech
        return text_to_speech(result['full_text'], api_url, tts_engine, tts_lang)
    
    return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Client for image processing and speech API')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000', 
                        help='Base URL of the image processing API (default: http://localhost:5000)')
    parser.add_argument('--tts-url', type=str, default='http://localhost:5001',
                        help='Base URL of the text-to-speech API (default: http://localhost:5001)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Remove text command
    remove_parser = subparsers.add_parser('remove-text', help='Remove text from an image')
    remove_parser.add_argument('image_path', type=str, help='Path to the input image')
    remove_parser.add_argument('--languages', type=str, default='en', 
                            help='Comma-separated list of language codes (e.g., "en,fr")')
    remove_parser.add_argument('--inpaint-radius', type=int, default=3, 
                            help='Radius for inpainting algorithm')
    
    # Extract text command
    extract_parser = subparsers.add_parser('extract-text', help='Extract text from an image')
    extract_parser.add_argument('image_path', type=str, help='Path to the input image')
    extract_parser.add_argument('--languages', type=str, default='en', 
                            help='Comma-separated list of language codes (e.g., "en,fr")')
    
    # Text to speech command
    speak_parser = subparsers.add_parser('speak', help='Convert text to speech')
    speak_parser.add_argument('text', type=str, help='Text to convert to speech')
    speak_parser.add_argument('--engine', type=str, default='gtts', choices=['gtts', 'pyttsx3'],
                            help='TTS engine to use')
    speak_parser.add_argument('--language', type=str, default='en', 
                            help='Language code for gTTS or voice ID for pyttsx3')
    speak_parser.add_argument('--rate', type=int, default=200, 
                            help='Speech rate (pyttsx3 only)')
    speak_parser.add_argument('--slow', action='store_true', 
                            help='Speak slowly (gTTS only)')
    
    # Image to speech command
    img2speech_parser = subparsers.add_parser('img2speech', help='Extract text from image and convert to speech')
    img2speech_parser.add_argument('image_path', type=str, help='Path to the input image')
    img2speech_parser.add_argument('--ocr-langs', type=str, default='en', 
                               help='Comma-separated list of language codes for OCR')
    img2speech_parser.add_argument('--tts-engine', type=str, default='gtts', choices=['gtts', 'pyttsx3'],
                                help='TTS engine to use')
    img2speech_parser.add_argument('--tts-lang', type=str, default='en', 
                               help='Language for TTS')
    
    # List available voices command
    voices_parser = subparsers.add_parser('list-voices', help='List available TTS voices')
    
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'remove-text':
        output_path = remove_text_from_image(
            args.image_path, 
            args.api_url, 
            args.languages, 
            args.inpaint_radius
        )
        
        # Display the original and processed images if available
        if output_path:
            try:
                import matplotlib.pyplot as plt
                
                original = Image.open(args.image_path)
                processed = Image.open(output_path)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                axes[0].imshow(original)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(processed)
                axes[1].set_title('Text Removed')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("Matplotlib not available for displaying images.")
    
    elif args.command == 'extract-text':
        extract_text_from_image(
            args.image_path, 
            args.api_url, 
            args.languages
        )
    
    elif args.command == 'speak':
        # Use the TTS API URL for speak command
        audio_path = text_to_speech(
            args.text, 
            args.tts_url, 
            args.engine, 
            args.language, 
            args.rate, 
            args.slow
        )
        
        # Try to play the audio if available
        if audio_path:
            try:
                import playsound
                print("Playing audio...")
                playsound.playsound(audio_path)
            except ImportError:
                print("playsound module not available. Install with 'pip install playsound' to play audio.")
    
    elif args.command == 'img2speech':
        # Extract text from image API
        result = extract_text_from_image(
            args.image_path,
            args.api_url,
            args.ocr_langs
        )
        
        if result and result.get('full_text'):
            # Convert text to speech using TTS API
            audio_path = text_to_speech(
                result['full_text'], 
                args.tts_url, 
                args.tts_engine, 
                args.tts_lang
            )
            
            # Try to play the audio if available
            if audio_path:
                try:
                    import playsound
                    print("Playing audio...")
                    playsound.playsound(audio_path)
                except ImportError:
                    print("playsound module not available. Install with 'pip install playsound' to play audio.")
    
    elif args.command == 'list-voices':
        # Get available voices from TTS API
        try:
            response = requests.get(f"{args.tts_url}/voices")
            if response.status_code == 200:
                voices = response.json()['voices']
                print("\nAvailable voices:")
                for i, voice in enumerate(voices):
                    print(f"{i+1}. {voice['name']} (ID: {voice['id']})")
            else:
                print(f"❌ Error: {response.text}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    else:
        parser.print_help() 