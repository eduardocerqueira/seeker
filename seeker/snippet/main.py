#date: 2024-05-13T16:43:46Z
#url: https://api.github.com/gists/43d235baa708a38ac02101cc5fc1b6d6
#owner: https://api.github.com/users/jericjan

from pathlib import Path
import re

song_folder = input('Paste your osu song folder here (Default: AppData/Local/osu!/Songs): ') or r"C:\Users\USER\AppData\Local\osu!\Songs"
song_folder = Path(song_folder)

output = ""

def generate_md(title, map_id):
    global output
    title = title.replace('[','\[').replace(']','\]')
    if map_id == '-1':
        output += f"No beatmap link for **{title}**  \n"
    else:    
        output += f"[{title}](https://osu.ppy.sh/beatmapsets/{map_id})  \n"

for file in song_folder.iterdir():
    if file.is_dir():
        
        res = re.findall(r'(^\d+) +(.+)', file.name)

        if res:
            map_id, title = res[0]
            generate_md(title, map_id)
        elif file.name == "Failed":
            pass
        else:
            for osu_file in file.glob("*.osu"): 
                with open(osu_file, encoding='utf-8') as f:
                    file_contents = f.read()
                map_id = re.findall(r'BeatmapSetID:(-?\d+)', file_contents)
                if map_id:
                    generate_md(file.name, map_id[0])
                        

with open("result.txt", "w") as f:
    f.write(output)

print("Done! check 'result.txt'")    