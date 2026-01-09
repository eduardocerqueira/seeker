#date: 2026-01-09T17:10:12Z
#url: https://api.github.com/gists/c1782613508793cfe62266db0a08d125
#owner: https://api.github.com/users/0xrabbyte

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

CONF = bytes.fromhex('1f11081b401f1102001f1137641b4e0d011f110b')
HEADER = bytes.fromhex('1d7630004800')
FOOTER = bytes.fromhex('1b64021b64021f1111')
FRESH = bytes.fromhex('1f1108')
CHUNK_SIZE = 420


def load_and_resize_grayscale(image_path: str, width: int = 576) -> Image.Image:
    img = Image.open(image_path).convert('L')
    height = max(1, int(img.height * width / img.width))
    return img.resize((width, height), Image.LANCZOS)


def binarize_image_fs(img_gray: Image.Image, threshold: int = 128, visual: bool = False) -> np.ndarray:
    """Floyd-Steinberg 抖动二值化，返回 uint8 (0/255) 图像。"""
    arr = np.array(img_gray, dtype=np.float32)
    h, w = arr.shape

    y_iter = range(h)
    if tqdm is not None:
        y_iter = tqdm(y_iter, total=h, desc='Dither (FS)', unit='row')

    for y in y_iter:
        for x in range(w):
            old = arr[y, x]
            new = 0.0 if old < threshold else 255.0
            err = old - new
            arr[y, x] = new
            if x + 1 < w:
                arr[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    arr[y + 1, x - 1] += err * 3 / 16
                arr[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    arr[y + 1, x + 1] += err * 1 / 16

    output = np.clip(arr, 0, 255).astype(np.uint8)
    if visual:
        plt.figure(figsize=(8, output.shape[0] * 8 / output.shape[1]))
        plt.imshow(output, cmap='gray', vmin=0, vmax=255)
        plt.title('Floyd-Steinberg Dithered')
        plt.axis('off')
        plt.show()
    return output


def bw_image_to_raster_bytes(bw_0_255: np.ndarray) -> bytes:
    """将 0/255 二值图按每 8 像素 -> 1 字节 (左高位) 打包；黑=1 白=0。"""
    rows, cols = bw_0_255.shape
    byte_list = []

    y_iter = range(rows)
    if tqdm is not None:
        y_iter = tqdm(y_iter, total=rows, desc='Pack raster', unit='row')

    for y in y_iter:
        for x in range(0, cols, 8):
            byte = 0
            for b in range(8):
                if x + b < cols:
                    bit = 0 if bw_0_255[y, x + b] > 0 else 1
                    byte |= (bit << (7 - b))
            byte_list.append(byte)
    return bytes(byte_list)


def pack_escpos_raster(
    bw_0_255: np.ndarray,
    density: int = 2,
    chunk_size: int = CHUNK_SIZE,
    header: bytes = HEADER,
    footer: bytes = FOOTER,
    conf: bytes = CONF,
    blank_lines: int = 0,
):
    """把二值化图像打包成: CONF + (HEADER + height_le + img_bytes + FOOTER)，再按 chunk_size 分包。"""
    if blank_lines < 0:
        raise ValueError(f"blank_lines 不能为负数，当前 blank_lines={blank_lines}")

    rows, cols = bw_0_255.shape
    if cols % 8 != 0:
        raise ValueError(f"width 必须是 8 的倍数，当前 width={cols}")

    if blank_lines:
        pad = np.full((blank_lines, cols), 255, dtype=np.uint8)
        if bw_0_255.dtype != np.uint8:
            bw_0_255 = bw_0_255.astype(np.uint8)
        bw_0_255 = np.vstack([bw_0_255, pad])
        rows += blank_lines

    img_bytes = bw_image_to_raster_bytes(bw_0_255)
    height_bytes = rows.to_bytes(2, 'little')
    data = header + height_bytes + img_bytes + footer

    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    conf_bytes = bytearray(conf)
    conf_bytes[8] = 2 * density
    chunks = [bytes(conf_bytes)] + chunks + [FRESH] * 4
    return chunks, rows, cols

def image_to_chunks(image_path, width=576, density=2, visual=False):
    """兼容入口：内部调用二值化 + 打包分包。"""
    img_gray = load_and_resize_grayscale(image_path, width=width)
    bw = binarize_image_fs(img_gray, visual=visual)
    return pack_escpos_raster(bw, density=density)


def write_chunks_txt(chunks, out_path: str) -> None:
    with open(out_path, 'w', encoding='utf-8') as f:
        chunk_iter = enumerate(chunks)
        if tqdm is not None:
            chunk_iter = tqdm(chunk_iter, total=len(chunks), desc='Write txt', unit='chunk')

        for idx, chunk in chunk_iter:
            f.write(f'Chunk {idx:02d} ({len(chunk)} bytes):\n')
            f.write(chunk.hex() + '\n\n')


def send_chunks_serial(
    chunks,
    port: str,
    baudrate: int,
    timeout: float,
):
    import serial

    with serial.Serial(
        port,
        baudrate,
        timeout=timeout
    ) as ser:
        chunk_iter = enumerate(chunks)
        if tqdm is not None:
            chunk_iter = tqdm(chunk_iter, total=len(chunks), desc='Send serial', unit='chunk')

        for idx, chunk in chunk_iter:
            ser.write(chunk)
            msg = f'已发送分包 {idx:02d}，长度 {len(chunk)} 字节。'
            if tqdm is None:
                print(msg)
            recv = ser.read(256)
            if recv:
                msg = f"分包 {idx:02d} 收到回复: {recv.hex()}"
                if tqdm is not None:
                    tqdm.write(msg)
                else:
                    print(msg)


def parse_args():
    p = argparse.ArgumentParser(
        description='input.jpg -> Floyd-Steinberg 二值化 -> ESC/POS 打包 -> 420 bytes 分包',
    )
    p.add_argument('--image', default='input.jpg', help='输入图片路径 (default: input.jpg)')
    p.add_argument('--threshold', type=int, default=128, help='二值化阈值 (default: 128)')
    p.add_argument('--density', type=int, default=2, help='打印密度参数 (default: 2)')
    p.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='分包大小 (default: 420)')
    p.add_argument('--blank-lines', type=int, default=48, help='打包时在图片底部追加的空行数 (default: 48)')
    p.add_argument('--visual', action='store_true', help='用 matplotlib 显示二值化结果')

    p.add_argument('--out-txt', default='out_chunks.txt', help='输出分包 hex 的 txt 文件路径')
    p.add_argument('--txt', action='store_true', help='不输出 txt')

    p.add_argument('--no-send-serial', action='store_true', help='发送到串口/蓝牙串口')
    p.add_argument('--port', default='COM5', help='串口号 (default: COM5)')
    p.add_argument('--baud', type=int, default=115200, help='波特率 (default: 115200)')
    p.add_argument('--timeout', type=float, default=0.6, help='读取超时秒 (default: 0.6)')
    return p.parse_args()


def main():
    args = parse_args()

    img_gray = load_and_resize_grayscale(args.image)
    bw = binarize_image_fs(img_gray, threshold=args.threshold, visual=args.visual)
    chunks, rows, cols = pack_escpos_raster(
        bw,
        density=args.density,
        chunk_size=args.chunk_size,
        blank_lines=args.blank_lines,
    )

    print(f'图片尺寸: {cols}x{rows}。')
    print(f'共生成 {len(chunks)} 个分包，每包最多{args.chunk_size}字节。')

    if args.txt:
        write_chunks_txt(chunks, args.out_txt)
        print(f'已输出: {args.out_txt}')

    if not args.no_send_serial:
        send_chunks_serial(
            chunks,
            port=args.port,
            baudrate=args.baud,
            timeout=args.timeout
        )

if __name__ == "__main__":
    main()
