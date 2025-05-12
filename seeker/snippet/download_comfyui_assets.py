#date: 2025-05-12T16:51:32Z
#url: https://api.github.com/gists/38be0a9abbdb63a2029f8fbd10538b6e
#owner: https://api.github.com/users/kenotron

'''
This script downloads ComfyUI assets, including the IPAdapter_plus custom node and various models.
It assumes you are running on Windows and places the files in a "comfyui" folder
within your user's Documents directory.
'''
import os
import subprocess
import requests
from tqdm import tqdm # For progress bar, ensure it's installed: pip install tqdm

# --- Configuration ---
USER_DOCUMENTS_DIR = os.path.join(os.path.expanduser("~"), "Documents")
COMFYUI_BASE_DIR = os.path.join(USER_DOCUMENTS_DIR, "comfyui")
CUSTOM_NODES_DIR = os.path.join(COMFYUI_BASE_DIR, "custom_nodes")
MODELS_DIR = os.path.join(COMFYUI_BASE_DIR, "models")
CLIP_VISION_DIR = os.path.join(MODELS_DIR, "clip_vision")
IPADAPTER_MODELS_DIR = os.path.join(MODELS_DIR, "ipadapter")
LORAS_DIR = os.path.join(MODELS_DIR, "loras")

IPADAPTER_PLUS_REPO_URL = "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"

# Models to download { "url": "target_filename_or_path_relative_to_model_dir" }
# Renaming is handled if a different filename is given after the last '/'
MODELS_TO_DOWNLOAD = {
    CLIP_VISION_DIR: [
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors", "name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors", "name": "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"},
        {"url": "https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/image_encoder/pytorch_model.bin", "name": "clip-vit-large-patch14-336.bin"},
    ],
    IPADAPTER_MODELS_DIR: [
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors", "name": "ip-adapter_sd15.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin", "name": "ip-adapter_sd15_light_v11.bin"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors", "name": "ip-adapter-plus_sd15.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors", "name": "ip-adapter-plus-face_sd15.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors", "name": "ip-adapter-full-face_sd15.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors", "name": "ip-adapter_sd15_vit-G.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors", "name": "ip-adapter_sdxl_vit-h.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", "name": "ip-adapter-plus_sdxl_vit-h.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors", "name": "ip-adapter-plus-face_sdxl_vit-h.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors", "name": "ip-adapter_sdxl.safetensors"},
        # FaceID models
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin", "name": "ip-adapter-faceid_sd15.bin"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin", "name": "ip-adapter-faceid-plusv2_sd15.bin"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait-v11_sd15.bin", "name": "ip-adapter-faceid-portrait-v11_sd15.bin"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl.bin", "name": "ip-adapter-faceid_sdxl.bin"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin", "name": "ip-adapter-faceid-plusv2_sdxl.bin"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl.bin", "name": "ip-adapter-faceid-portrait_sdxl.bin"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-portrait_sdxl_unnorm.bin", "name": "ip-adapter-faceid-portrait_sdxl_unnorm.bin"},
        # Community models
        {"url": "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sd15.safetensors", "name": "ip_plus_composition_sd15.safetensors"},
        {"url": "https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors", "name": "ip_plus_composition_sdxl.safetensors"},
        {"url": "https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/ip_adapter_plus_general.bin?download=true", "name": "Kolors-IP-Adapter-Plus.bin"},
        {"url": "https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-FaceID-Plus/resolve/main/ipa-faceid-plus.bin?download=true", "name": "Kolors-IP-Adapter-FaceID-Plus.bin"},
    ],
    LORAS_DIR: [
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors", "name": "ip-adapter-faceid_sd15_lora.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15_lora.safetensors", "name": "ip-adapter-faceid-plusv2_sd15_lora.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sdxl_lora.safetensors", "name": "ip-adapter-faceid_sdxl_lora.safetensors"},
        {"url": "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", "name": "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"},
    ]
}

# --- Helper Functions ---
def create_directory_if_not_exists(directory_path):
    '''Creates a directory if it doesn't already exist.'''
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def download_file(url, destination_path, file_name):
    '''Downloads a file from a URL to a destination path with a progress bar.'''
    full_path = os.path.join(destination_path, file_name)
    if os.path.exists(full_path):
        print(f"File already exists: {full_path}")
        return

    print(f"Downloading {file_name} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))

        with open(full_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        print(f"Successfully downloaded: {full_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {url}: {e}")

def clone_repo(repo_url, destination_path):
    '''Clones a git repository.'''
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    target_repo_path = os.path.join(destination_path, repo_name)

    if os.path.exists(target_repo_path):
        print(f"Repository {repo_name} already exists in {destination_path}. Skipping clone.")
        # Optionally, you could add logic here to pull latest changes
        # subprocess.run(["git", "pull"], cwd=target_repo_path, check=True)
        return

    print(f"Cloning {repo_url} into {destination_path}...")
    try:
        subprocess.run(["git", "clone", repo_url, target_repo_path], check=True, capture_output=True, text=True)
        print(f"Successfully cloned {repo_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository {repo_url}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        print("Please ensure git is installed and in your system's PATH.")
    except FileNotFoundError:
        print("Error: Git command not found. Please ensure git is installed and in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred while cloning {repo_url}: {e}")

# --- Main Execution ---
def main():
    print("Starting ComfyUI IPAdapter Plus setup...")

    # 1. Create base directories
    print("\n--- Creating Directories ---")
    create_directory_if_not_exists(COMFYUI_BASE_DIR)
    create_directory_if_not_exists(CUSTOM_NODES_DIR)
    create_directory_if_not_exists(MODELS_DIR)
    create_directory_if_not_exists(CLIP_VISION_DIR)
    create_directory_if_not_exists(IPADAPTER_MODELS_DIR)
    create_directory_if_not_exists(LORAS_DIR)

    # 2. Clone ComfyUI_IPAdapter_plus repository
    print("\n--- Cloning Repository ---")
    clone_repo(IPADAPTER_PLUS_REPO_URL, CUSTOM_NODES_DIR)

    # 3. Download models
    print("\n--- Downloading Models ---")
    for directory, models in MODELS_TO_DOWNLOAD.items():
        print(f"\nDownloading models for: {directory}")
        create_directory_if_not_exists(directory) # Ensure sub-directory exists
        for model_info in models:
            download_file(model_info["url"], directory, model_info["name"])

    print("\n-----------------------------------------")
    print("ComfyUI IPAdapter Plus setup script finished.")
    print(f"All files should be located in: {COMFYUI_BASE_DIR}")
    print("Please ensure you have ComfyUI installed separately.")
    print("You may need to install insightface for FaceID models: pip install insightface")
    print("Refer to the IPAdapter_plus documentation for further setup if needed.")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()
