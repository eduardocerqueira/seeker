#date: 2023-11-30T16:48:23Z
#url: https://api.github.com/gists/4dea727cececb01b65c61f46b4a4dac2
#owner: https://api.github.com/users/sagarjauhari

# Run Llamafile with LLaVA locally
# Source: https://news.ycombinator.com/item?id=38465645

# 1. Download the 4.26GB llamafile-server-0.1-llava-v1.5-7b-q4 file from https://huggingface.co/jartine/llava-v1.5-7B-GGUF/blob/main/...:
wget https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llamafile-server-0.1-llava-v1.5-7b-q4

# 2. Make that binary executable, by running this in a terminal:
chmod 755 llamafile-server-0.1-llava-v1.5-7b-q4

# 3. Run your new executable, which will start a web server on port 8080:
./llamafile-server-0.1-llava-v1.5-7b-q4

# 4. Navigate to http://127.0.0.1:8080/ to upload an image and start chatting with the model about it in your browser.
# Screenshot here: https://simonwillison.net/2023/Nov/29/llamafile/
open "http://127.0.0.1:8080/"