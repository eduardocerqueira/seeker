#date: 2023-07-04T16:49:11Z
#url: https://api.github.com/gists/04328cf8fe1fb1517db769feb2d2f977
#owner: https://api.github.com/users/imwhocodes

import cv2
import numpy as np
import depthai as dai

# Weights to use when blending depth/rgbWarped image (should equal 1.0)
rgbWeight = 0.2
depthWeight = 0.8


UNWARP_ALPHA = 1

msgs = dict()

def add_msg(msg, name, seq = None):
    if seq is None:
        seq = msg.getSequenceNum()
    seq = str(seq)
    if seq not in msgs:
        msgs[seq] = dict()
    msgs[seq][name] = msg

def get_msgs():
    global msgs
    seq_remove = [] # Arr of sequence numbers to get deleted
    for seq, syncMsgs in msgs.items():
        seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
        # Check if we have both detections and color frame with this sequence number
        if len(syncMsgs) == 3: # rgbWarped + depth
            for rm in seq_remove:
                del msgs[rm]
            return syncMsgs # Returned synced msgs
    return None

def updateBlendWeights(percent_rgb):
    """
    Update the rgbWarped and depth weights used to blend depth/rgbWarped image
    @param[in] percent_rgb The rgbWarped weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight


# Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
# Otherwise (False), the aligned depth is automatically upscaled to 1080p
downscaleColor = False
fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution  = dai.MonoCameraProperties.SensorResolution. THE_720_P
colorResolution = dai.ColorCameraProperties.SensorResolution.THE_720_P


def getUndistortMap(calibData, ispSize):
    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, ispSize[0], ispSize[1]))
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB))

    R1 = None   #np.identity(3)

    M2, _= cv2.getOptimalNewCameraMatrix(
                                                        M1,
                                                        d1,
                                                        ispSize,
                                                        UNWARP_ALPHA,
                                                        ispSize,
                                                        False
                                                    )


    return cv2.initUndistortRectifyMap(M1, d1, R1, M2, ispSize, cv2.CV_32FC1)


def getMesh(mapX, mapY):
                                                    
    meshCellSize = 16
    mesh0 = []
    # Creates subsampled mesh which will be loaded on to device to undistort the image
    for y in range(mapX.shape[0] + 1): # iterating over height of the image
        if y % meshCellSize == 0:
            rowLeft = []
            for x in range(mapX.shape[1]): # iterating over width of the image
                if x % meshCellSize == 0:
                    if y == mapX.shape[0] and x == mapX.shape[1]:
                        rowLeft.append(mapX[y - 1, x - 1])
                        rowLeft.append(mapY[y - 1, x - 1])
                    elif y == mapX.shape[0]:
                        rowLeft.append(mapX[y - 1, x])
                        rowLeft.append(mapY[y - 1, x])
                    elif x == mapX.shape[1]:
                        rowLeft.append(mapX[y, x - 1])
                        rowLeft.append(mapY[y, x - 1])
                    else:
                        rowLeft.append(mapX[y, x])
                        rowLeft.append(mapY[y, x])
            if (mapX.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)

            mesh0.append(rowLeft)

    mesh0 = np.array(mesh0)
    meshWidth = mesh0.shape[1] // 2
    meshHeight = mesh0.shape[0]
    mesh0.resize(meshWidth * meshHeight, 2)

    mesh = list(map(tuple, mesh0))

    return mesh, meshWidth, meshHeight

def create_pipeline(calibData):
    # Create pipeline
    pipeline = dai.Pipeline()
    queueNames = []

    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    rgbRawOut       = pipeline.create(dai.node.XLinkOut)
    rgbWarpedOut    = pipeline.create(dai.node.XLinkOut)
    disparityOut    = pipeline.create(dai.node.XLinkOut)

    rgbRawOut.setStreamName("rgbRaw")
    queueNames.append("rgbRaw")
    rgbWarpedOut.setStreamName("rgbWarped")
    queueNames.append("rgbWarped")
    disparityOut.setStreamName("disp")
    queueNames.append("disp")

    #Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(colorResolution)
    camRgb.setFps(fps)
    camRgb.setPreviewSize(camRgb.getIspSize())
    camRgb.setInterleaved(False)
    lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
    if lensPosition:
        camRgb.initialControl.setManualFocus(lensPosition)

    mapX, mapY = getUndistortMap(calibData, camRgb.getPreviewSize())

    manip = pipeline.create(dai.node.Warp)
    mesh, meshWidth, meshHeight = getMesh(mapX, mapY)
    manip.setWarpMesh(mesh, meshWidth, meshHeight)
    manip.setMaxOutputFrameSize(camRgb.getPreviewHeight() * camRgb.getPreviewWidth() * 3)
    camRgb.preview.link(manip.inputImage)

    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # LR-check is required for depth alignment
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setAlphaScaling(UNWARP_ALPHA)

    # Linking
    camRgb.preview.link(rgbRawOut.input)
    manip.out.link(rgbWarpedOut.input)
    stereo.disparity.link(disparityOut.input)


    left.out.link(stereo.left)
    right.out.link(stereo.right)



    return pipeline, mapX, mapY

# Connect to device and start pipeline
with dai.Device() as device:
    calibData = device.readCalibration2()
    pipeline, mapX, mapY = create_pipeline(calibData)
    device.startPipeline(pipeline)

    blendedWarpedWindowName = "rgbDeviceWarped-depth"
    cv2.namedWindow(blendedWarpedWindowName)
    cv2.createTrackbar('RGB Weight %', blendedWarpedWindowName, int(rgbWeight*100), 100, updateBlendWeights)

    blendedRawWindowName = "rgbHostWarped-depth"
    cv2.namedWindow(blendedRawWindowName)
    cv2.createTrackbar('RGB Weight %', blendedRawWindowName, int(rgbWeight*100), 100, updateBlendWeights)





    while True:
        for name in ('rgbRaw', 'rgbWarped', 'disp'):
            msg = device.getOutputQueue(name).tryGet()
            if msg is not None:
                add_msg(msg, name)

        synced = get_msgs()

        if synced:

            frameRgbRaw = synced["rgbRaw"].getCvFrame()
            frameRgbDeviceWarped = synced["rgbWarped"].getCvFrame()
            frameDisp = synced["disp"].getFrame()


            maxDisparity = 95
            # Optional, extend range 0..95 -> 0..255, for a better visualisation
            if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
            # Optional, apply false colorization
            if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
            frameDisp = np.ascontiguousarray(frameDisp)

            # Need to have both frames in BGR format before blending
            if len(frameDisp.shape) < 3:
                frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)


            blended = cv2.addWeighted(frameRgbDeviceWarped, rgbWeight, frameDisp, depthWeight, 0)
            cv2.imshow(blendedWarpedWindowName, blended)

            frameRgbHostWarped = cv2.remap(frameRgbRaw, mapX, mapY, cv2.INTER_LINEAR)
            blended = cv2.addWeighted(frameRgbHostWarped, rgbWeight, frameDisp, depthWeight, 0)
            cv2.imshow(blendedRawWindowName, blended)


        if cv2.waitKey(10) == ord('q'):
            break