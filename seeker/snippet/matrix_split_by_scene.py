#date: 2021-12-15T17:19:19Z
#url: https://api.github.com/gists/8cc6f4dba5d0d13b316f1dbf093908a7
#owner: https://api.github.com/users/tomasonjo

article_clean_txt = "\n".join([line for line in article_txt.split("\n") if not "THE MATRIX" in line and not "CONTINUED" in line])

#Split by scenes
scenes = []
single_scene = []
for line in article_clean_txt.split("\n"):
  # If empty line
  if not line or line.startswith("OMITTED"):
    continue
  if line.startswith("INT.") or line.startswith("EXT.") or line.startswith("THE END"):
    scenes.append(("\n").join(single_scene))
    single_scene = []
  single_scene.append(line)
 
# Extract scene names
scene_names = [el.split("\n")[0] for el in scenes]