#date: 2024-03-06T17:09:21Z
#url: https://api.github.com/gists/8dc4023b953a5f4825251aed8d8bf5ef
#owner: https://api.github.com/users/TheDeepHub

# Load the mask image (the same mask used earlier)
mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

# Capture video from a file or camera (replace 'your_video.mp4' with your video file or use 0 for webcam)
video_capture = cv2.VideoCapture(video_path)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab a frame")
        break

    # Use the updated function that includes model predictions
    frame_with_predictions = draw_bounding_boxes_and_predict(frame, mask, model)

    cv2.imshow('Video with Parking Slot Occupancy', frame_with_predictions)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the capture when everything is done
video_capture.release()
cv2.destroyAllWindows()