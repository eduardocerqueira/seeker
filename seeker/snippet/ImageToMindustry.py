#date: 2023-11-28T17:01:49Z
#url: https://api.github.com/gists/3dc434834e07f24affd41b14fcb48ae5
#owner: https://api.github.com/users/rafa-br34

import pyperclip
import math
from PIL import Image


c_MaxColorDifference = 16 # 4, 8, 16
c_TargetResolution = 80
c_SliceOperations = 998 # 998
c_DrawsPerFlush = 70
c_MaxCubeSize = 40
c_TargetImage = "image.png"
c_Output = "display1"

def CompareColor(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2 + (A[2] - B[2]) ** 2)

def FLAG(X, Y):
    return ((Y & 0xFFFF) << 16) | ((X & 0xFFFF) << 0)

def CheckBlock(Pixels, SizeX, SizeY, StartX, StartY):
    MainColor = Pixels[StartX, StartY]

    for S in range(c_MaxCubeSize):
        if StartX + S >= SizeX or StartY + S >= SizeY:
            return S
        
        V = 0
        for X in range(S):
            for Y in range(S):
                V += CompareColor(MainColor, Pixels[StartX + X, StartY + Y])

        if V > (c_MaxColorDifference * S):
            return S

    return c_MaxCubeSize


from PIL import ImageDraw

def main():
    OriginalTargetImage = Image.open(c_TargetImage).convert("RGB").rotate(180, expand=True)
    TargetImage = OriginalTargetImage.resize((c_TargetResolution, c_TargetResolution))

    DoneList = set()
    PXS = TargetImage.load()

    DrawOperations = []

    for Y in range(c_TargetResolution):
        for X in range(c_TargetResolution):
            if FLAG(X, Y) in DoneList:
                continue
            Size = CheckBlock(PXS, c_TargetResolution, c_TargetResolution, X, Y)

            for BX in range(Size):
                for BY in range(Size):
                    DoneList.add(FLAG(X + BX, Y + BY))
            
            DrawOperations.append([PXS[X, Y], (X, Y), Size])

        if Y % c_MaxCubeSize == 0:
            DoneList.clear()

    DrawOperations.sort(key=lambda V: (V[0][0] << 16) | (V[0][1] << 8) | (V[0][2] << 0))

    Operations = []
    LastColor = None

    #DebugImage = Image.new("RGB", (c_TargetResolution, c_TargetResolution))
    #DebugCanvas = ImageDraw.Draw(DebugImage)

    
    r = ""
    i = 0
    
    def OP(String, Count=0):
        nonlocal r, i
        r += String + '\n'
        i += Count

    def CopyPart(Part):
        print(Part, end=''); pyperclip.copy(Part)
        input("Copied Part, Press Key To Proceed")

    for Operation in DrawOperations:
        [Color, [X, Y], Size] = Operation

        CL = f"draw color {Color[0]} {Color[1]} {Color[2]} 255 0 0"
        RE = f"draw rect {X} {Y} {Size} {Size} 0 0 {c_Output}"
        DF = f"drawflush {c_Output}"

        if Color != LastColor:
            OP(CL, 1)
            LastColor = Color
        
        if c_SliceOperations > 0 and i >= c_SliceOperations:
            OP(f"{DF}\nend", 2)
            CopyPart(r); r = ""
            OP(CL, 1)
            i = 1

        if i % c_DrawsPerFlush == 0:
            OP(DF, 1)

        #DebugCanvas.rectangle((X, Y, X+Size, Y+Size), fill=(Color[0] << 0) | (Color[1] << 8) | (Color[2] << 16))
        OP(RE, 1)
    
    OP(f"{DF}\nend", 2)
    CopyPart(r)
    #DebugImage.save("DebugImage.png")




if __name__ == "__main__":
    main()