#date: 2022-09-06T16:55:22Z
#url: https://api.github.com/gists/43619e4d73d08e36589b328e6b29d0cd
#owner: https://api.github.com/users/vadimkantorov

# download only auto subtitles in lang=ru, resources:
# https://superuser.com/questions/927523/how-to-download-only-subtitles-of-videos-using-youtube-dl
# http://ytdl-org.github.io/youtube-dl/download.html

# wget https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe && ./vcredist_x86.exe /install /quiet /norestart
# wget https://yt-dl.org/downloads/latest/youtube-dl.exe

# downloadonlysubtitles 'https://www.youtube.com/watch?v=VBHhhGOGdKk'

alias downloadonlysubtitles='youtube-dl --write-auto-sub --sub-lang ru --skip-download' # or ./youtube-dl.exe on Windows