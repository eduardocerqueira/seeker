#date: 2025-07-07T16:55:10Z
#url: https://api.github.com/gists/1f72786c48153f1b977e68b69c571a81
#owner: https://api.github.com/users/phoenixthrush

"""
Copyright (c) 2025 phoenixthrush

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""

import cv2
import numpy as np
import os
import sys


class VideoThumbnailMaker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Store dimensions for reuse
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video loaded: {os.path.basename(video_path)}")
        print(
            f"Duration: {self.duration:.2f}s, Frames: {self.total_frames}, FPS: {self.fps:.2f}")

    def extract_thumbnails(self, num_thumbnails=20, min_grid_height=1600):
        """Extract thumbnails from video at evenly spaced intervals."""
        if self.total_frames <= 0:
            return []

        # Calculate thumbnail dimensions for minimum grid height
        rows, cols, padding = 4, 5, 10
        required_thumb_height = (
            min_grid_height - (rows + 1) * padding) // rows
        aspect_ratio = self.original_width / self.original_height
        thumbnail_height = required_thumb_height
        thumbnail_width = int(required_thumb_height * aspect_ratio)

        # Extract thumbnails
        thumbnails = []
        interval = max(1, self.total_frames // num_thumbnails)
        for i in range(num_thumbnails):
            frame_number = min(i * interval, self.total_frames - 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = self.cap.read()
            if ret:
                thumbnail = cv2.resize(
                    frame, (thumbnail_width, thumbnail_height))
                timestamp = frame_number / self.fps if self.fps > 0 else 0
                thumbnail = self.add_timestamp_overlay(thumbnail, timestamp)
                thumbnails.append(thumbnail)

        return thumbnails

    def add_timestamp_overlay(self, image, timestamp):
        """Add timestamp overlay to top right of thumbnail."""
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        overlay_image = image.copy()
        height, width = overlay_image.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        padding = 10

        (text_width, text_height), _ = cv2.getTextSize(
            time_str, font, font_scale, font_thickness)

        x = width - text_width - padding
        y = text_height + padding

        # Semi-transparent background
        bg_x1, bg_y1 = x - 3, y - text_height - 3
        bg_x2, bg_y2 = x + text_width + 3, y + 3

        overlay = overlay_image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, overlay_image, 0.3, 0, overlay_image)

        cv2.putText(overlay_image, time_str, (x, y), font,
                    font_scale, (255, 255, 255), font_thickness)

        return overlay_image

    def create_thumbnail_grid(self, thumbnails, rows=4, cols=5, padding=10):
        """Create grid layout with banner showing video statistics."""
        if not thumbnails:
            return None

        thumb_height, thumb_width = thumbnails[0].shape[:2]

        # Get video file information
        file_name = os.path.basename(self.video_path)
        try:
            file_size_mb = os.path.getsize(self.video_path) / (1024 * 1024)
            file_size_str = f"{file_size_mb:.1f} MB"
        except:
            file_size_str = "?"

        # Get video resolution and FPS
        try:
            if self.original_width <= 0 or self.original_height <= 0:
                resolution_str = "?"
            else:
                fps_str = f"{self.fps:.1f}" if self.fps > 0 else "?"
                resolution_str = f"{self.original_width}x{self.original_height} / {fps_str} fps"
        except:
            resolution_str = "?"

        # Format duration
        try:
            if self.duration > 0:
                duration_hours = int(self.duration // 3600)
                duration_minutes = int((self.duration % 3600) // 60)
                duration_seconds = int(self.duration % 60)
                duration_str = f"{duration_hours:02d}:{duration_minutes:02d}:{duration_seconds:02d}"
            else:
                duration_str = "?"
        except:
            duration_str = "?"

        # Calculate dimensions
        grid_width = cols * thumb_width + (cols + 1) * padding
        grid_height = rows * thumb_height + (rows + 1) * padding
        banner_height = 150
        banner_padding = 15

        # Create banner
        banner_image = np.zeros((banner_height, grid_width, 3), dtype=np.uint8)
        banner_image.fill(45)

        # Add banner text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)
        font_thickness = 2
        line_spacing = 30

        # Left column - labels
        labels = ["File Name:", "File Size:", "Resolution:", "Duration:"]
        # Right column - values
        values = [file_name, file_size_str, resolution_str, duration_str]

        # Calculate position for values column (pre-calculate font metrics)
        max_label_width = max(cv2.getTextSize(label, font, font_scale, font_thickness)[
                              0][0] for label in labels)
        values_x_position = banner_padding + max_label_width + 30

        # Draw all text in one loop
        for i, (label, value) in enumerate(zip(labels, values)):
            y_pos = banner_padding + (i + 1) * line_spacing - 10
            cv2.putText(banner_image, label, (banner_padding, y_pos),
                        font, font_scale, font_color, font_thickness)
            cv2.putText(banner_image, value, (values_x_position, y_pos),
                        font, font_scale, font_color, font_thickness)

        # Add title and author at top right
        title_text = "VIDEO THUMBNAIL MAKER"
        author_text = "by phoenixthrush"

        title_font_scale = 1.2
        title_font_thickness = 3
        author_font_scale = 0.8
        author_font_thickness = 2
        author_color = (200, 200, 200)

        (title_width, title_height), _ = cv2.getTextSize(
            title_text, font, title_font_scale, title_font_thickness)
        (author_width, author_height), _ = cv2.getTextSize(
            author_text, font, author_font_scale, author_font_thickness)

        title_x = grid_width - title_width - banner_padding
        title_y = banner_padding + title_height + 5
        author_x = grid_width - author_width - banner_padding
        author_y = title_y + author_height + 18

        cv2.putText(banner_image, title_text, (title_x, title_y),
                    font, title_font_scale, font_color, title_font_thickness)
        cv2.putText(banner_image, author_text, (author_x, author_y),
                    font, author_font_scale, author_color, author_font_thickness)

        # Create complete image
        total_height = banner_height + grid_height
        complete_image = np.zeros(
            (total_height, grid_width, 3), dtype=np.uint8)
        complete_image[0:banner_height, 0:grid_width] = banner_image

        # Create grid section
        grid_section = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        grid_section.fill(30)

        # Place thumbnails with white borders
        for i, thumbnail in enumerate(thumbnails):
            if i >= rows * cols:
                break

            row = i // cols
            col = i % cols

            x = col * (thumb_width + padding) + padding
            y = row * (thumb_height + padding) + padding

            # Draw white border
            cv2.rectangle(grid_section,
                          (x - 1, y - 1),
                          (x + thumb_width + 1, y + thumb_height + 1),
                          (255, 255, 255), 1)

            grid_section[y:y+thumb_height, x:x+thumb_width] = thumbnail

        complete_image[banner_height:total_height, 0:grid_width] = grid_section

        return complete_image

    def __del__(self):
        """Clean up video capture object."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_file_path>")
        return

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    try:
        # Create thumbnail maker
        maker = VideoThumbnailMaker(video_path)

        # Generate thumbnails
        thumbnails = maker.extract_thumbnails(num_thumbnails=20)

        if not thumbnails:
            print("Error: Could not extract thumbnails from the video.")
            return

        # Create grid layout
        grid_image = maker.create_thumbnail_grid(thumbnails, rows=4, cols=5)

        # Save the result
        output_path = "video_thumbnails.png"
        cv2.imwrite(output_path, grid_image)

        print(f"Thumbnail saved as '{output_path}'")

        # Display the result
        cv2.imshow("Video Thumbnails", grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
