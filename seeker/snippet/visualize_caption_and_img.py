#date: 2024-01-01T16:44:52Z
#url: https://api.github.com/gists/9131e502b4f99402fa52834f9e630d51
#owner: https://api.github.com/users/ShunyuYao

import glob
import os

from PIL import ImageOps, Image, ImageFont, ImageDraw
import json
import math

def visualize_caption_in_img(img_path, text_caption_path, line_break_words_num=15):
    img = Image.open(img_path).convert('RGB')
    
    with open(text_caption_path) as f:
        lines = f.readlines()
        text_caption = ' '.join([line.strip() for line in lines])
        text_caption = text_caption.strip()
    
    text_prompts = text_caption.split(' ')
    new_text_prompt = ''
    for i in range(len(text_prompts)):
        new_text_prompt += text_prompts[i]
        if i < len(text_prompts) - 1:
            new_text_prompt += ' '
        if i % line_break_words_num == 0 and i > 0:
            new_text_prompt += '\n'

    padding_text = int(50 * math.ceil(len(text_prompts) / line_break_words_num))

    img = ImageOps.expand(img, border=(0, padding_text, 0, 0), fill='white')
    ttf = ImageFont.truetype(font='/usr/share/fonts/truetype/tlwg/Umpush.ttf', size=30)
    img_draw = ImageDraw.Draw(img)
    img_draw.text((10, 5), new_text_prompt, font=ttf, fill='black')
    return img