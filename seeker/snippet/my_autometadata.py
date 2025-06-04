#date: 2025-06-04T17:09:22Z
#url: https://api.github.com/gists/ff00449ffd01d10808d731c2734206ae
#owner: https://api.github.com/users/Cdaprod

import os
import cv2
import xml.etree.ElementTree as ET

def get_video_technical_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0
    cap.release()
    return {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
    }

def blur_score(path, sample_frame=10):
    cap = cv2.VideoCapture(path)
    ret, frame = False, None
    for _ in range(sample_frame):
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def scene_brightness(path, sample_frame=10):
    cap = cv2.VideoCapture(path)
    ret, frame = False, None
    for _ in range(sample_frame):
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return 0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()

def make_description(tech, blur, brightness):
    desc_parts = []
    if tech["fps"] > 45:
        desc_parts.append("slow motion")
    if blur > 100:
        desc_parts.append("sharp focus")
    else:
        desc_parts.append("soft focus")
    if brightness > 140:
        desc_parts.append("bright")
    elif brightness < 60:
        desc_parts.append("dark")
    else:
        desc_parts.append("neutral lighting")
    desc_parts.append(f"{tech['width']}x{tech['height']} resolution")
    return " ".join(desc_parts).capitalize() + " stock footage."

def create_sidecar_xml(video_path, metadata):
    base = os.path.splitext(video_path)[0]
    xml_path = base + ".xml"

    root = ET.Element("video")
    for key, val in metadata.items():
        ET.SubElement(root, key).text = str(val)
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Generated: {xml_path}")

def process_dir(directory):
    for file in os.listdir(directory):
        if not file.lower().endswith(".mp4"):
            continue
        path = os.path.join(directory, file)

        # 1. Extract technical info
        tech = get_video_technical_info(path)
        blur = blur_score(path)
        brightness = scene_brightness(path)

        # 2. Generate description & keywords
        description = make_description(tech, blur, brightness)
        slowmo = tech["fps"] > 45
        keywords = [
            "stock",
            "footage",
            "slow motion" if slowmo else "normal speed",
            "sharp" if blur > 100 else "soft",
            "bright" if brightness > 140 else "neutral" if brightness > 60 else "dark",
            f"{tech['width']}x{tech['height']}",
        ]

        # 3. Build metadata dictionary
        metadata = {
            "filename": os.path.basename(path),
            "absolute_path": os.path.abspath(path),
            "description": description,
            "keywords": ", ".join(keywords),
            "fps": tech["fps"],
            "duration": round(tech["duration"], 2),
            "resolution": f"{tech['width']}x{tech['height']}",
            "blur_score": round(blur, 2),
            "brightness": round(brightness, 2),
            "slow_motion": str(slowmo),
        }

        # 4. Create or update sidecar XML
        create_sidecar_xml(path, metadata)

if __name__ == "__main__":
    # Run this script from the directory containing your .mp4 files, e.g.:
    # cd B:\Video\StockFootage\Batches\well_pump
    current_dir = os.getcwd()
    process_dir(current_dir)