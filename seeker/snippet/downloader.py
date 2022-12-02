#date: 2022-12-02T17:04:29Z
#url: https://api.github.com/gists/dc2607a137ebb3272c2887547c477c06
#owner: https://api.github.com/users/Shayan-Raza

from pytube import YouTube

link = input("Enter the link of the Youtube Video:" )

def download_video (link) : 
  yt = YouTube(link)
  yt = yt.streams.get_highest_resolution()
  
  try : 
    yt.download()
  except : 
    print("Error Occured")
    
  print("Video Saved")
  
download_video(link)