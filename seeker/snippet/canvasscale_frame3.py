#date: 2023-06-23T16:57:44Z
#url: https://api.github.com/gists/a8aaccb80f95eb1dd4b736abbcd71550
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
            print(objects_dict)
            # scale_all2(canvas, 0, 0, scale_factor, scale_factor)
            for i in objects_dict:
                resize_frame(objects_dict[i][1], scale_factor)
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
            print(objects_dict)
            for i in objects_dict:
                resize_frame(objects_dict[i][1], scale_factor)
            # scale_all2(canvas, 0, 0, scale_factor, scale_factor)
    for item in canvas.find_all():
        check_element_size(canvas, item)

_drag_data = {}
objects_dict = {}
def start_drag(event):
    global _drag_data
    if str(event.widget).startswith(".!canvas.!frame"):
        _drag_data = {
            'start_x': event.x_root,
            'start_y': event.y_root,
        }
    else:
        canvas.scan_mark(event.x, event.y)


def drag(event):
    if str(event.widget).startswith(".!canvas.!frame"):
        dx = event.x_root - _drag_data['start_x']
        dy = event.y_root - _drag_data['start_y']
        canvas.move(objects_dict[str(event.widget)][0], dx, dy)
        _drag_data['start_x'] = event.x_root
        _drag_data['start_y'] = event.y_root
    else:
        canvas.scan_dragto(event.x, event.y, gain=1)
root = tk.Tk()

scale_factor = 1.0

canvas = tk.Canvas(root, width=400, height=400, background="purple")
canvas.pack(expand=1, fill="both")

def create_frame2(event):
    canvas = event.widget
    x = canvas.canvasx(event.x)  # Adjust for scroll position
    y = canvas.canvasy(event.y)
    frame = tk.Frame(canvas, width=200, height=200, bg="white")
    frame.propagate(False)
    item_id = canvas.create_window(x, y, anchor='nw', window=frame)
    text_widget = tk.Text(frame)
    text_widget.pack()

    objects_dict[str(text_widget).split(" ")[0]] = item_id, frame

canvas.create_rectangle(50, 50, 200, 200, fill="red")
canvas.create_rectangle(100, 100, 300, 200, fill='red')
canvas.create_oval(150, 150, 250, 250, fill='blue')
canvas.create_rectangle(50, 50, 200, 200, fill='red')

root.bind("<MouseWheel>", zoom)
root.bind("<Button-4>", zoom)
root.bind("<Button-5>", zoom)
canvas.bind('<Button-3>', create_frame2)
canvas.bind_all("<Button-1>", start_drag)
canvas.bind_all("<B1-Motion>", drag)
def check_element_size(canvas, element_id):
    x1, y1, x2, y2 = canvas.bbox(element_id)
    width = x2 - x1
    height = y2 - y1
    print("Element size:", width, "x", height, element_id)
root.mainloop()
