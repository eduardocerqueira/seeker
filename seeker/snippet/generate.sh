#date: 2023-08-07T16:58:57Z
#url: https://api.github.com/gists/123a30dee1fb02751cd78a0b10338a36
#owner: https://api.github.com/users/greg-randall

# a few notes, 
# first, you'll need to install edge-tts (pip3 install edge-tts) and opus (sudo apt-get install opus-tools) 
# second, you'll have to clean up your book's text file before generating an audiobook. for the below example
# i'm using the time machine (https://www.gutenberg.org/files/35/35-0.txt) the file needs it's short lines 
# unwrapped calibre's (https://calibre-ebook.com/) conversion utility has a very good line unwrapper under 
# 'heuristics'. you may also want to remove extra text, chapter listings, index, glossary, etc.


mkdir audiobook
cd audiobook

csplit -f book_split_ -s -b %05d.txt -z ../the_time_machine.txt /\n/ {*}

find . -maxdepth 1 -name '*.txt' -exec edge-tts -f {} --voice en-US-SteffanNeural --write-media {}_Steffan.mp3 \;

mkdir text-split
mv *.txt text-split

mkdir audio-files
mv *.mp3 audio-files

cd audio-files

find * -type f > mp3-list.txt
sed -e 's/^/file /' -i mp3-list.txt
ffmpeg -f concat -i mp3-list.txt ../full.wav

cd ..

opusenc --downmix-mono --bitrate 32 --vbr full.wav full.opus

mv full.opus ../the_time_machine.opus