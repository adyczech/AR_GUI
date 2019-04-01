from freetype import *
import pickle

FACENAME = "Nasalization_Bold.ttf"
PIXEL_HEIGHT = 116
FILE_NAME = "Nasalization_Bold_42.pickle"

font_data = []

def next_p2(num):
    """ If num isn't a power of 2, will return the next higher power of two """
    rval = 1
    while rval < num:
        rval <<= 1
    return rval


def make_dlist(ft, ch):
    ft.load_char(chr(ch))
    bitmap = ft.glyph.bitmap

    width = next_p2(bitmap.width)
    height = next_p2(bitmap.rows)

    expanded_data = []
    for j in range(height):
        for i in range(width):
            if (i >= bitmap.width) or (j >= bitmap.rows):
                expanded_data.append(0)
                expanded_data.append(0)
            else:
                value = bitmap.buffer[j * bitmap.width + i]
                expanded_data.append(value)
                expanded_data.append(value)
    font_data.append(expanded_data)


ft = Face(FACENAME)
ft.set_char_size(height=PIXEL_HEIGHT * 64, vres=90)
for i in range(383):
    make_dlist(ft, i)
    print(i)

fileObject = open(FILE_NAME, 'wb')
pickle.dump(font_data, fileObject)
fileObject.close()

print(len(font_data))