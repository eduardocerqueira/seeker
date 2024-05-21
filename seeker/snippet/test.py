#date: 2024-05-21T16:47:37Z
#url: https://api.github.com/gists/7809981cf919e34dc0a33a9fac48e00d
#owner: https://api.github.com/users/zebraed

import maya.cmds as cmds

# 名前を設定
pointA = 'pointA'
pointB = 'pointB'
locator = 'locator1'

# ポイントAとポイントBのノードが存在するか確認
if not cmds.objExists(pointA):
    cmds.error(f"{pointA} does not exist.")
if not cmds.objExists(pointB):
    cmds.error(f"{pointB} does not exist.")
if not cmds.objExists(locator):
    locator = cmds.spaceLocator(name='locator1')[0]

# ノードの作成
pointA_matrixMult = cmds.createNode('pointMatrixMult', name='pointA_matrixMult')
pointB_matrixMult = cmds.createNode('pointMatrixMult', name='pointB_matrixMult')
vector_AB = cmds.createNode('plusMinusAverage', name='vector_AB')
vectorNormalize = cmds.createNode('vectorProduct', name='vectorNormalize')
vectorLength = cmds.createNode('multiplyDivide', name='vectorLength')
extendVector = cmds.createNode('multiplyDivide', name='extendVector')
newPosition = cmds.createNode('plusMinusAverage', name='newPosition')
locatorPosition = cmds.createNode('pointMatrixMult', name='locatorPosition')

# ポイントAとポイントBの座標取得
cmds.connectAttr(f'{pointA}.translate', f'{pointA_matrixMult}.inPoint')
cmds.connectAttr(f'{pointA}.parentMatrix[0]', f'{pointA_matrixMult}.inMatrix')

cmds.connectAttr(f'{pointB}.translate', f'{pointB_matrixMult}.inPoint')
cmds.connectAttr(f'{pointB}.parentMatrix[0]', f'{pointB_matrixMult}.inMatrix')

# ベクトルの計算
cmds.connectAttr(f'{pointB_matrixMult}.output', f'{vector_AB}.input3D[0]')
cmds.connectAttr(f'{pointA_matrixMult}.output', f'{vector_AB}.input3D[1]')
cmds.setAttr(f'{vector_AB}.operation', 2)  # Subtract vectors

# ベクトルの正規化
cmds.connectAttr(f'{vector_AB}.output3D', f'{vectorNormalize}.input1')
cmds.setAttr(f'{vectorNormalize}.operation', 0)  # No operation needed, just normalize
cmds.setAttr(f'{vectorNormalize}.normalizeOutput', True)

# ベクトルの延長線上のポイント計算
cmds.setAttr(f'{vectorLength}.input1X', 10.0)  # 延長する長さを設定
cmds.connectAttr(f'{vectorNormalize}.output', f'{vectorLength}.input2X')

cmds.connectAttr(f'{pointA_matrixMult}.output', f'{newPosition}.input3D[0]')
cmds.connectAttr(f'{vectorLength}.outputX', f'{extendVector}.input1X')
cmds.connectAttr(f'{vectorNormalize}.outputX', f'{extendVector}.input2X')
cmds.setAttr(f'{extendVector}.operation', 1)  # Scale the vector

cmds.connectAttr(f'{pointA_matrixMult}.output', f'{newPosition}.input3D[0]')
cmds.connectAttr(f'{extendVector}.outputX', f'{newPosition}.input3D[1].input3Dx')
cmds.connectAttr(f'{extendVector}.outputY', f'{newPosition}.input3D[1].input3Dy')
cmds.connectAttr(f'{extendVector}.outputZ', f'{newPosition}.input3D[1].input3Dz')
cmds.setAttr(f'{newPosition}.operation', 1)  # Add vectors

# ロケーターノードを拘束
cmds.connectAttr(f'{newPosition}.output3D', f'{locatorPosition}.inPoint')
cmds.connectAttr(f'{locator}.parentMatrix[0]', f'{locatorPosition}.inMatrix')
cmds.connectAttr(f'{locatorPosition}.output', f'{locator}.translate')