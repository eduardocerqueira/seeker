#date: 2025-02-12T17:10:57Z
#url: https://api.github.com/gists/3672eb69bbeeceaccc246d60af852531
#owner: https://api.github.com/users/tejashah88

# Copied from https://github.com/tejashah88/blueos-oakd-extension/blob/5e8a4bdcd46e6bc60e3f3c501464d8046789abe6/src/oakd_pipeline.py

# Create pipeline
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

videoEnc = pipeline.create(dai.node.VideoEncoder)
rgbEncOut = pipeline.create(dai.node.XLinkOut)
rgbEncOut.setStreamName('rgb')
videoEnc.setDefaultProfilePreset(25, dai.VideoEncoderProperties.Profile.H264_MAIN)
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(rgbEncOut.input)

# Create left/right mono cameras for Stereo depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera('left')

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera('right')

# Create a node that will produce the depth map
depth = pipeline.create(dai.node.StereoDepth)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(True)
depth.setExtendedDisparity(True)
# NOTE: Subpixel disparity is of UINT16 format, which is unsupported by VideoEncoder
depth.setSubpixel(False)
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

# Colormap
colormap = pipeline.create(dai.node.ImageManip)
colormap.initialConfig.setColormap(dai.Colormap.TURBO, depth.initialConfig.getMaxDisparity())
colormap.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)

videoEnc = pipeline.create(dai.node.VideoEncoder)
# Depth resolution/FPS will be the same as mono resolution/FPS
videoEnc.setDefaultProfilePreset(monoLeft.getFps(), dai.VideoEncoderProperties.Profile.H264_HIGH)

# Link
depth.disparity.link(colormap.inputImage)
colormap.out.link(videoEnc.input)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName('depth')
videoEnc.bitstream.link(xout.input)