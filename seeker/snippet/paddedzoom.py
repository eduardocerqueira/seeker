#date: 2023-11-17T17:00:49Z
#url: https://api.github.com/gists/50327f249c43e2f1b3a0309c8404a438
#owner: https://api.github.com/users/repogiu

import cv2

def paddedzoom(img, zoomfactor=0.8):
    
    '''
    Zoom in/out an image while keeping the input image shape.
    i.e., zero pad when factor<1, clip out when factor>1.
    there is another version below (paddedzoom2)
    '''

    out  = np.zeros_like(img)
    zoomed = cv2.resize(img, None, fx=zoomfactor, fy=zoomfactor)
    
    h, w = img.shape
    zh, zw = zoomed.shape
    
    if zoomfactor<1:    # zero padded
        out[(h-zh)/2:-(h-zh)/2, (w-zw)/2:-(w-zw)/2] = zoomed
    else:               # clip out
        out = zoomed[(zh-h)/2:-(zh-h)/2, (zw-w)/2:-(zw-w)/2]

    return out
    
    
def paddedzoom2(img, zoomfactor=0.8):
    
    ' does the same thing as paddedzoom '
    
    h,w = img.shape
    M = cv2.getRotationMatrix2D( (w/2,h/2), 0, zoomfactor) 
    
    return cv2.warpAffine(img, M, img.shape[::-1])
    
