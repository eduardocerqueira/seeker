#date: 2023-06-30T17:00:14Z
#url: https://api.github.com/gists/a85b86f84dd86c582bc8f2041757aa78
#owner: https://api.github.com/users/gmmoreira

yt-dlp -v --ignore-config --split-chapters -o 'chapter:%(title)s - %(section_number)02d - %(section_title)s.%(ext)s' -f bestaudio[ext=m4a] --extract-audio --embed-thumbnail 'https://youtube.com/watch?v=aaaaaa'