from PIL import Image
import numpy as np
from OpenGL.GLU import *
from OpenGL.GL import *
import cv2
from src.AR_Picture import *
from src.CreateFont import *

class HUDInfo:

    def __init__(self, parent, label, scale=1):
        self.label = label.replace('\\n', '\n').replace('\\t', '\t')
        self.scale = scale
        self.parent = parent

        self.nol = self.label.count("\n")  # Number Of Lines

        self.info_top = ARPicture('content/hud/HUD_info_top.png', cv2.IMREAD_UNCHANGED)
        self.info_middle = ARPicture('content/hud/HUD_info_middle.png', cv2.IMREAD_UNCHANGED)
        self.info_bottom = ARPicture('content/hud/HUD_info_bottom.png', cv2.IMREAD_UNCHANGED)

        self.font = self.parent.nasa_font_12

    def draw_info(self, view_matrix):
        self.info_top.set_modelview_matrix(view_matrix, self.parent.win_width/2, self.parent.win_height/2 + self.info_top.iy/2, 0)
        self.info_top.draw()

        for i in range(0, self.nol):
            self.info_middle.set_modelview_matrix(view_matrix, self.parent.win_width/2, self.parent.win_height/2 - (i*self.info_middle.iy + self.info_middle.iy/2), 0)
            self.info_middle.draw()

        self.info_bottom.set_modelview_matrix(view_matrix, self.parent.win_width/2, self.parent.win_height/2 - (self.nol*self.info_middle.iy + self.info_bottom.iy/2), 0)
        self.info_bottom.draw()

        glLoadIdentity()
        glColor3fv(self.parent.font_color)
        CreateFont.glPrint(self.font, self.parent.win_width/2 - 425, self.parent.win_height/2 - 18, 0, self.label, None, True, self.scale)
        glColor3f(1.0, 1.0, 1.0)

    def change_labels(self, label):
        self.label = label.replace('\\n', '\n').replace('\\t', '\t')
        self.nol = self.label.count("\n")  # Number Of Lines
