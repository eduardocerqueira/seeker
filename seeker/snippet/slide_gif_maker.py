#date: 2025-09-25T16:56:04Z
#url: https://api.github.com/gists/9a739aa66be9ef74d68a6c8aba750548
#owner: https://api.github.com/users/itsk1mlot

import os
import zipfile
from PIL import Image, ImageFilter

# ===== 설정 =====
zip_path = "ddtemp.zip"         # 네 ZIP 파일 경로
extract_dir = "ddtemp_extract"  # 압축 풀 폴더
output_path = "billboard.gif"   # 출력 GIF 경로
target_size = (5120, 1080)      # 최종 GIF 크기
panel_count = 10                # 한 화면에 표시될 패널 수
frames_per_step = 30            # 프레임 수 (높을수록 부드럽게, 용량 커짐)
duration_per_frame = int(2000 / frames_per_step)  # 2초 / frames_per_step

# ===== 압축 해제 =====
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# ===== 이미지 전처리 =====
panel_w = target_size[0] // panel_count
panel_h = target_size[1]

panels = []
for file in sorted(os.listdir(extract_dir)):
    path = os.path.join(extract_dir, file)
    try:
        img = Image.open(path).convert("RGB")
    except:
        continue

    img_ratio = img.width / img.height
    target_ratio = panel_w / panel_h
    if img_ratio > target_ratio:
        new_height = panel_h
        new_width = int(new_height * img_ratio)
    else:
        new_width = panel_w
        new_height = int(new_width / img_ratio)

    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 블러 배경 생성
    bg = resized.copy().resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(40))

    # 중앙 배치
    x = (panel_w - resized.width) // 2
    y = (panel_h - resized.height) // 2
    bg.paste(resized, (x, y))

    panels.append(bg)

# ===== 프레임 생성 =====
frames = []
n = len(panels)

for shift in range(n):
    for substep in range(frames_per_step):
        offset = int((substep / frames_per_step) * panel_w)
        frame = Image.new("RGB", target_size, (0, 0, 0))

        for i in range(panel_count + 1):
            idx = (shift + i) % n
            frame.paste(panels[idx], (i * panel_w - offset, 0))

        frames.append(frame)

# ===== GIF 저장 =====
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=duration_per_frame,
    loop=0,
    optimize=True
)

# gif를 mp4로 바꾸는 방법
# ffmpeg -i billboard.gif -movflags faststart -pix_fmt yuv420p -vf "scale=5120:1080:flags=lanczos" billboard.mp4