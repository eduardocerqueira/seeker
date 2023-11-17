#date: 2023-11-17T16:33:41Z
#url: https://api.github.com/gists/653b1fb6d0b0731236b63060a039fd7d
#owner: https://api.github.com/users/bstee615

#!/bin/bash
pip install huggingface-hub==0.19.0

git clone https://github.com/ggerganov/llama.cpp
(cd llama.cpp; make -j -k)

huggingface-cli download TheBloke/Nous-Capybara-34B-GGUF nous-capybara-34b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
./llama.cpp/main -ngl 32 -m nous-capybara-34b.Q4_K_M.gguf --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -i -ins

# Interact with the LLM through the command-line chat interface. Have fun!
