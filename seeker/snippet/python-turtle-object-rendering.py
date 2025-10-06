#date: 2025-10-06T16:46:58Z
#url: https://api.github.com/gists/fe383a90e09c3f372b59cd00d2cfdb7a
#owner: https://api.github.com/users/phaze-the-dumb

from turtle import *
import math
import time

i = 0

fps = 60
fov = 60 * ( math.pi / 180 )

class transform:
    def __init__( self ):
        self.yaw = 0
        self.pitch = 100 * ( math.pi / 180 )

        self.x = 0
        self.y = 0
        self.z = 25

    def GetBasisVectors( self ):
        ihat_yaw = float3(math.cos(self.yaw), 0, math.sin(self.yaw))
        jhat_yaw = float3(0, 1, 0)
        khat_yaw = float3(-math.sin(self.yaw), 0, math.cos(self.yaw))

        ihat_pitch = float3(1, 0, 0)
        jhat_pitch = float3(0, math.cos(self.pitch), -math.sin(self.pitch))
        khat_pitch = float3(0, math.sin(self.pitch), math.cos(self.pitch))

        ihat = TransformVector(ihat_yaw, jhat_yaw, khat_yaw, ihat_pitch)
        jhat = TransformVector(ihat_yaw, jhat_yaw, khat_yaw, jhat_pitch)
        khat = TransformVector(ihat_yaw, jhat_yaw, khat_yaw, khat_pitch)

        return ( ihat, jhat, khat )
    
def TransformVector( ihat, jhat, khat, v ):
    return float3(
        (ihat.x * v.x) + (jhat.x * v.y) + (khat.x * v.z),
        (ihat.y * v.x) + (jhat.y * v.y) + (khat.y * v.z),
        (ihat.z * v.x) + (jhat.z * v.y) + (khat.z * v.z),
    )

class float3:
    def __init__( self, x, y, z ):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return "(" + str(self.x) + " " + str(self.y) + " " + str(self.z) + ")"

class tri:
    def __init__( self, p1, p2, p3 ):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

class obj:
    def __init__( self, path ):
        self.tris = []
        self.transform = transform()

        file = open(path, "r")
        lines = file.readlines()

        vertex = []

        for line in lines:
            if line.startswith("v "):
                point = line.replace("v ", '').split(' ')

                if point[0] == '':
                    point.remove('')

                vertex.append(float3(float(point[0]), float(point[1]), float(point[2])))
            elif line.startswith("f "):
                face = line.replace("f ", '').split(' ')
                self.tris.append(tri(
                    vertex[int(face[0].split('/')[0]) - 1],
                    vertex[int(face[1].split('/')[0]) - 1],
                    vertex[int(face[2].split('/')[0]) - 1],
                ))


    def toWorldSpace( self, point ):
        ( ihat, jhat, khat ) = self.transform.GetBasisVectors()
        point = TransformVector(ihat, jhat, khat, point)

        point.x += self.transform.x
        point.y += self.transform.y
        point.z += self.transform.z

        return point

object = obj("Sting-Sword-lowpoly.obj")

# object = obj([
#     tri(
#         float3(-1, 3, -1),
#         float3(1, 3, -1),
#         float3(1, 3, -3)
#     ),
#     tri(
#         float3(-1, 3, -1),
#         float3(1, 3, -3),
#         float3(-1, 3, -3)
#     ),

#     tri(
#         float3(-1, 1, -1),
#         float3(1, 1, -1),
#         float3(1, 3, -1)
#     ),
#     tri(
#         float3(-1, 1, -1),
#         float3(1, 3, -1),
#         float3(-1, 3, -1)
#     ),


#     tri(
#         float3(-1, -1, 1),
#         float3(1, -1, 1),
#         float3(1, 1, 1)
#     ),
#     tri(
#         float3(-1, -1, 1),
#         float3(1, 1, 1),
#         float3(-1, 1, 1)
#     ),

#     tri(
#         float3(-1, 1, -1),
#         float3(1, 1, -1),
#         float3(1, 1, 1)
#     ),
#     tri(
#         float3(-1, 1, -1),
#         float3(1, 1, 1),
#         float3(-1, 1, 1)
#     ),

#     tri(
#         float3(-1, -1, 3),
#         float3(1, -1, 3),
#         float3(1, -1, 1)
#     ),
#     tri(
#         float3(-1, -1, 3),
#         float3(1, -1, 1),
#         float3(-1, -1, 1)
#     ),
# ])

# object = obj([
#     tri(
#         float3(-1, -1, 1),
#         float3(1, -1, 1),
#         float3(1, 1, 1)
#     ),
#     tri(
#         float3(-1, -1, 1),
#         float3(1, 1, 1),
#         float3(-1, 1, 1)
#     ),

#     tri(
#         float3(1, -1, -1),
#         float3(1, -1, 1),
#         float3(1, 1, 1)
#     ),
#     tri(
#         float3(1, -1, -1),
#         float3(1, 1, 1),
#         float3(1, 1, -1)
#     ),


#     tri(
#         float3(-1, -1, -1),
#         float3(1, -1, -1),
#         float3(1, 1, -1)
#     ),
#     tri(
#         float3(-1, -1, -1),
#         float3(1, 1, -1),
#         float3(-1, 1, -1)
#     ),

#     tri(
#         float3(-1, -1, -1),
#         float3(-1, -1, 1),
#         float3(-1, 1, 1)
#     ),
#     tri(
#         float3(-1, -1, -1),
#         float3(-1, 1, 1),
#         float3(-1, 1, -1)
#     ),


#     tri(
#         float3(-1, 1, -1),
#         float3(1, 1, -1),
#         float3(1, 1, 1)
#     ),
#     tri(
#         float3(-1, 1, -1),
#         float3(1, 1, 1),
#         float3(-1, 1, 1)
#     ),

#     tri(
#         float3(-1, -1, -1),
#         float3(1, -1, -1),
#         float3(1, -1, 1)
#     ),
#     tri(
#         float3(-1, -1, -1),
#         float3(1, -1, 1),
#         float3(-1, -1, 1)
#     ),
# ])

def WorldToScreenSpace( point ):
    point = object.toWorldSpace(point)

    screenheight = math.tan(fov / 2) * 2
    pixelsPerWorldUnit = screensize()[1] / screenheight / point.z

    return ( point.x * pixelsPerWorldUnit, point.y * pixelsPerWorldUnit )

while True:
    tracer(0)
    speed(0)
    ht()
    up()

    object.transform.yaw += math.pi * 0.025
    clear()

    for treyangle in object.tris:
        p1 = WorldToScreenSpace(treyangle.p1)
        p2 = WorldToScreenSpace(treyangle.p2)
        p3 = WorldToScreenSpace(treyangle.p3)

        # fillcolor(0, 0, 0)

        goto(p1[0], p1[1])
        down()
        # begin_fill()
        goto(p2[0], p2[1])
        goto(p3[0], p3[1])
        goto(p1[0], p1[1])
        # end_fill()
        up()

    i += 1
    update()
    time.sleep(1 / fps)