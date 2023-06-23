#date: 2023-06-23T16:36:57Z
#url: https://api.github.com/gists/8c2332dbe2bfba9ca388b5f1c050fd97
#owner: https://api.github.com/users/secemp9

import tkinter as tk

def resize_frame(frame, scale):
    frame.config(width=frame.winfo_reqwidth() * scale, height=frame.winfo_reqheight() * scale)
    for child in frame.winfo_children():
        w = child.winfo_reqwidth() * scale
        h = child.winfo_reqheight() * scale
        try:
            child.config(width=w, height=h)
        except tk.TclError:
            pass

count_zoomin = 0
count_zoomout = 0
def zoom(event):
    global scale_factor, count_zoomin, count_zoomout
    n = 4
    if event.num == 4 or event.delta == 120:
        if count_zoomin == n or count_zoomin > n:
            pass
        elif count_zoomin < n:
            # factor = 1.001 ** event.delta
            count_zoomin += 1
            count_zoomout -= 1
            print("in", count_zoomin, "out", count_zoomout)
            # print(event.delta)
            # scale_factor = 0.01 - event.delta
            scale_factor = 1.001 ** event.delta
            canvas.scale("all", 0, 0, scale_factor, scale_factor)
            resize_frame(frame, scale_factor)
    elif event.num == 5 or event.delta == -120:
        if count_zoomout == n or count_zoomout > n:
            pass
        elif count_zoomout < n:
            # factor = 1.001 ** event.delta
            count_zoomout += 1
            count_zoomin -= 1
            print("in", count_zoomin, "out", count_zoomout)
            # scale_factor = 0.01 * event.delta
            scale_factor = 1.001 ** event.delta
            canvas.scale("all", 0, 0, scale_factor, scale_factor)
            resize_frame(frame, scale_factor)
    for item in canvas.find_all():
        check_element_size(canvas, item)


root = tk.Tk()

scale_factor = 1.0

canvas = tk.Canvas(root, width=400, height=400, background="black")
canvas.pack(expand=1, fill="both")

frame = tk.Frame(canvas, width=200, height=200, bg="blue")
frame.propagate(False)
frame_id = canvas.create_window(200, 200, window=frame)
text_widget = tk.Text(frame)
text_widget.pack()

root.bind("<MouseWheel>", zoom)
root.bind("<Button-4>", zoom)
root.bind("<Button-5>", zoom)
def check_element_size(canvas, element_id):
    x1, y1, x2, y2 = canvas.bbox(element_id)
    width = x2 - x1
    height = y2 - y1
    print("Element size:", width, "x", height, element_id)
root.mainloop()
