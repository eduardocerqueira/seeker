#date: 2022-02-23T16:53:20Z
#url: https://api.github.com/gists/8b70f3f9878d2df6c3f4ee984780c954
#owner: https://api.github.com/users/ecdedios

# for manipulating the PDF
import fitz

# for OCR using PyTesseract
import cv2                              # pre-processing images
import pytesseract                      # extracting text from images
import numpy as np
import matplotlib.pyplot as plt         # displaying output images

from PIL import Image