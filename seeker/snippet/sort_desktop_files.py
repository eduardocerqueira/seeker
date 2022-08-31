#date: 2022-08-31T16:52:12Z
#url: https://api.github.com/gists/f8d7cb05eeebae5ec157a36ca42d5a06
#owner: https://api.github.com/users/SKaplanOfficial

import PyXA
app = PyXA.Application("System Events")

desktop_files = app.desktop_folder.files()
desktop_folders = app.desktop_folder.folders()

# Create sorting bin folders
images_folder = app.make("folder", {"name": "Images"})
videos_folder = app.make("folder", {"name": "Videos"})
audio_folder = app.make("folder", {"name": "Audio"})
documents_folder = app.make("folder", {"name": "Documents"})
desktop_folders.push(images_folder, videos_folder, audio_folder, documents_folder)

# Sort images
image_predicate = "name ENDSWITH '.png' OR name ENDSWITH '.jpg' OR name ENDSWITH '.jpeg' OR name ENDSWITH '.aiff'"
image_files = desktop_files.filter(image_predicate)
image_files.move_to(images_folder)

# Sort videos
video_predicate = "name ENDSWITH '.mov' OR name ENDSWITH '.mp4' OR name ENDSWITH '.avi' OR name ENDSWITH '.m4v'"
video_files = desktop_files.filter(video_predicate)
video_files.move_to(videos_folder)

# Sort audio
audio_predicate = "name ENDSWITH '.mp3' OR name ENDSWITH '.ogg'"
audio_files = desktop_files.filter(audio_predicate)
audio_files.move_to(audio_folder)

# Sort remaining (documents)
desktop_files.move_to(documents_folder)