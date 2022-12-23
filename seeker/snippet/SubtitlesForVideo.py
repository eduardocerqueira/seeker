#date: 2022-12-23T16:32:28Z
#url: https://api.github.com/gists/bb11fe80de086f56ef4c61a6f567030a
#owner: https://api.github.com/users/TruongVuGoBrrrrr

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api.formatters import WebVTTFormatter
from tkinter.ttk import *
from tkinter import *
import tkinter
import webvtt

#Cửa sổ giao diện
top = Tk()
top.geometry('600x600')
top.title("Tạo phụ đề cho video Youtube")

lbl = Label(top, text="ID của video:", font=("Arial", 20))
lbl.grid(column=0, row=0, sticky = E)

lbl = Label(top, text="Ngôn ngữ ban đầu:", font=("Arial", 20))
lbl.grid(column=0, row=1, sticky = E)

lbl = Label(top, text="Ngôn ngữ yêu cầu:", font=("Arial", 20))
lbl.grid(column=0, row=2, sticky = E)

txtid = Entry(top, width=20, font=("Arial", 20))
txtid.grid(column=1,row=0)

combo1 = Combobox(top, width=19, font=("Arial", 20))
combo1['value'] = ("vi","en","es","th","zh-Hans")
combo1.grid(column=1,row=1)

combo2 = Combobox(top, width=19, font=("Arial", 20))
combo2['value'] = ("vi","en","es","th","zh-Hans")
combo2.grid(column=1,row=2)

ketqua = Text(top, width=30, height=20, font=("Arial", 13))
ketqua.grid(column=1,row=4,sticky= W)

def main():
    vid_id = txtid.get()
    base_lang = combo1.get()
    wanted_lang = combo2.get()

    # Tạo file phụ đề ban đầu
    transcripts = YouTubeTranscriptApi.list_transcripts(vid_id)
    base_obj = transcripts.find_transcript([base_lang])
    base_tran = base_obj.fetch()

    fmt = TextFormatter()
    base_txt = fmt.format_transcript(base_tran)
    print("Writing {} Transcript ...".format(base_lang), end="")
    with open("transcripts/{}_transcript.txt".format(base_lang), "w" , encoding='utf-8') as f:
        f.write(base_txt)
    print("DONE")

    #Dịch sang ngôn ngữ khác
    if base_obj.is_translatable:
        wanted_tran = base_obj.translate(wanted_lang).fetch()
    else:
        print("CAN NOT translate transcript to {}".format(wanted_lang))
        quit()


    #Tạo phụ đề đã dịch
    wanted_txt = fmt.format_transcript(wanted_tran)
    print("Writing {} Transcript ...".format(wanted_lang), end="")
    with open("transcripts/{}_transcript.txt".format(wanted_lang), "w" , encoding='utf-8') as f:
        f.write(wanted_txt)
    print("DONE")

    # Tạo phụ đề hoàn chỉnh (có kèm thời gian)
    fmt = WebVTTFormatter()
    wanted_subs = fmt.format_transcript(wanted_tran)
    print("Writing {} Subtitles ...".format(wanted_lang), end="")
    with open("Subtitles/{}_subs.vtt".format(wanted_lang), "w" , encoding='utf-8') as f:
        f.write(wanted_subs)
    print("DONE")
    ketqua.insert(END, wanted_subs)





btn = Button(top, command=main , font=("Arial", 20), text="Tạo phụ đề", bg="white", fg="black")
btn.grid(column=0, row=4, sticky=N)

p1 = PhotoImage(file = 'youtubeicon.png')
top.iconphoto(False, p1)


top.mainloop()

