#date: 2023-12-13T17:02:24Z
#url: https://api.github.com/gists/e542580aad54154b8733a5f363f3c248
#owner: https://api.github.com/users/tyvsmith

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import google.generativeai as genai
import google.ai.generativelanguage as glm

API_KEY="YOUR_KEY_HERE" # add your key here
model = genai.GenerativeModel('gemini-pro-vision')

def capture_frame():
    # Capture a frame from the webcam
    ret, frame = webcam.read()

    # Increment the frame counter and update the text field
    global frame_counter
    
    # Display the frame in the UI
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_img = Image.fromarray(frame)
    global img
    img = resize_image(orig_img, max_width=1000)
    frame = ImageTk.PhotoImage(img)
    frame_label.configure(image=frame)
    frame_label.image = frame  # Keep a reference to avoid garbage collection


    # Schedule the next frame capture
    frame_label.after(30, capture_frame)  # 10 seconds in milliseconds

def set_text(text):
    # Update the text in the text field
    text_field.configure(text=text)

def resize_image(image, max_width):

    original_width, original_height = image.size
    if original_width > max_width:
        # Calculate the new height maintaining the aspect ratio
        height = int((max_width / original_width) * original_height)
        return image.resize((max_width, height), Image.ADAPTIVE)
    return image

def explain_image():
    root.after(0, update_explanation, "Loading...")
    response = model.generate_content(["Creatively and humorously make fun of the subject's physical appearance using only what's observed", img], stream=True)
    response.resolve()
    root.after(0, update_explanation, response.text)


def update_explanation(text):
    text_field.configure(text=f"Roast Machine: {text}")

genai.configure(api_key=API_KEY)
# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Create the UIQq
root = tk.Tk()
root.title("Roast Machine")

# Create the frame label
frame_label = tk.Label(root)
frame_label.grid(row=0, column=0)

# Create the text field
text_field = tk.Label(root, text="Roasting: ...", wraplength=400)
text_field.grid(row=0, column=1)

button = tk.Button(root, text="Roast Machine!", command=lambda: threading.Thread(target=explain_image).start())

button.grid(row=0, column=1, sticky="s")

# Schedule the first frame capture
frame_counter = 0
frame_label.after(0, capture_frame)  # 10 seconds in milliseconds

# Start the UI loop
root.mainloop()

# Cleanup
webcam.release()