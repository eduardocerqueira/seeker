#date: 2025-09-01T17:02:13Z
#url: https://api.github.com/gists/d5fecb885f0ec1206c090ad9c177cf7d
#owner: https://api.github.com/users/davidyang0168

#!/bin/bash

# --------------------------
# 配置变量
# --------------------------
DEVICE_ID="6010f51a"  # 目标设备的 adb 设备 ID，可通过 `adb devices` 查看
SCRCPY_DIR="/Users/kohane/scrcpy-macos-x86_64-v3.3.1"
ADB="$SCRCPY_DIR/adb"      # adb 可执行文件路径
SCRCPY="$SCRCPY_DIR/scrcpy"  # scrcpy 可执行文件路径

# scrcpy 参数配置
SCRCPY_MAX_SIZE=1024       # 最大分辨率（宽或高），屏幕较大可增大，单位 px
SCRCPY_VIDEO_BITRATE="2M"  # 视频比特率，单位 bit/s，可选值如 1M, 2M, 4M
SCRCPY_AUDIO_BITRATE="64k" # 音频比特率，可选 32k, 64k, 128k 等
SCRCPY_MAX_FPS=30          # 最大帧率，可选值如 15, 30, 60
SCRCPY_WINDOW_TITLE="MAA-Scrcpy" # scrcpy 窗口标题，方便识别窗口

# --------------------------
# 收尾函数
# --------------------------
cleanup() {
    echo "恢复设备分辨率和 DPI..."
    # 将设备分辨率恢复默认
    "$ADB" -s "$DEVICE_ID" shell wm size reset
    # 将设备 DPI 恢复默认
    "$ADB" -s "$DEVICE_ID" shell wm density reset
    echo "关闭 scrcpy..."
    # 杀掉后台运行的 scrcpy
    pkill -f "$SCRCPY"
}
# trap 捕获脚本退出信号，确保异常退出也会调用 cleanup
trap cleanup EXIT

# --------------------------
# 启动逻辑
# --------------------------
echo "设置设备分辨率和 DPI..."
# 设置分辨率为 1080x1920，可选值为任意符合屏幕比例的分辨率
"$ADB" -s "$DEVICE_ID" shell wm size 1080x1920
# 设置 DPI 为 360，可选值根据手机屏幕和适配需求调整
"$ADB" -s "$DEVICE_ID" shell wm density 360

echo "启动 scrcpy（后台运行，输出显示在终端）..."
"$SCRCPY" --serial "$DEVICE_ID" \       # 指定设备 ID
    --max-size "$SCRCPY_MAX_SIZE" \    # 限制显示分辨率，单位 px，保持性能
    --video-bit-rate "$SCRCPY_VIDEO_BITRATE" \  # 视频比特率，影响画质和带宽
    --audio-bit-rate "$SCRCPY_AUDIO_BITRATE" \  # 音频比特率
    --max-fps "$SCRCPY_MAX_FPS" \      # 最大帧率，控制 CPU/GPU 占用
    --window-title "$SCRCPY_WINDOW_TITLE" \  # 窗口标题
    --stay-awake \                      # 防止手机屏幕息屏
    --turn-screen-off &                 # 启动 scrcpy 时手机屏幕可选择关闭
# 可选 scrcpy 参数（根据需求）：
#   --fullscreen          全屏显示
#   --always-on-top       窗口总在最上层
#   --rotation            手动设置旋转角度 0, 90, 180, 270
#   --crop WxH+X+Y        裁剪显示区域
#   --record file.mp4     直接录屏保存到文件

echo "启动 MAA..."
# 打开 macOS GUI 应用程序 MAA
open -a "/Applications/MAA.app"

# --------------------------
# 等待 MAA GUI 退出
# --------------------------
echo "等待 MAA 退出中..."
while osascript -e 'application "MAA" is running' | grep -q true; do
    sleep 2
done

echo "MAA 已退出，执行收尾逻辑..."
