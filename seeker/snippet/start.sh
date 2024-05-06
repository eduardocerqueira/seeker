#date: 2024-05-06T17:08:18Z
#url: https://api.github.com/gists/c0e4286274d61ee3431a7b8616c4b450
#owner: https://api.github.com/users/xidecs

#!/bin/bash
for filename in ./drive/MyDrive/Dataset/ml/*.mp3; do
    echo "Processing" $(basename $filename)
    insanely-fast-whisper --file-name "$filename" --transcript-path "./${filename: "**********":(-4)}.json" --hf-token "$HF_TOKEN" --model-name "openai/whisper-large-v3" --diarization_model "pyannote/speaker-diarization-3.1"
done
