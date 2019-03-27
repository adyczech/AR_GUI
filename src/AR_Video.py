import time
import numpy as np
from threading import Thread
import cv2
from OpenGL.GLU import *
from OpenGL.GL import *
from PIL import Image
from pygame import mixer


class ARVideo:

    def __init__(self, video, sound=None):
        self.video = cv2.VideoCapture(video)
        self.isPlaying = False
        self.cur_frame = np.uint8()
        self.gl_texture = glGenTextures(1)
        self.music = sound
        self.is_displayed = False
        self.last_draw_time = time.time()
        if not isinstance(self.music, type(None)):
            mixer.pre_init(44100, -16, 1, 512)
            mixer.init()
            self.sound = mixer.Sound(sound)

    def start(self):
        Thread(target=self.play, args=(), daemon=True).start()

    def play(self):
        self.isPlaying = True
        nof = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_counter = 0
        if not isinstance(self.music, type(None)):
            self.sound.play()
        while True:
            if not self.is_displayed and (time.time() - self.last_draw_time >= 0.2):
                self.sound.set_volume(0)
            else:
                self.sound.set_volume(1)
            self.is_displayed = False

            if frame_counter == nof:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.isPlaying = False
                break

            ret, self.cur_frame = self.video.read()
            frame_counter += 1
            time.sleep(1/(fps*1.5))

    def get_current_frame(self):
        return self.cur_frame

    def draw_video(self, view_matrix, width, height):
        if not self.isPlaying:
            self.start()

        self.is_displayed = True
        self.last_draw_time = time.time()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        frame = self.get_current_frame()

        if np.size(frame) == 1:
            return

        # convert image to OpenGL texture format
        frame = Image.fromarray(frame)
        ix = frame.size[0]
        iy = frame.size[1]
        frame = frame.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.gl_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     frame)  # specify a two-dimensional texture image
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)  # set texture parameters(target, parameter name, parameter value)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        glPushMatrix()
        glLoadMatrixd(view_matrix)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)  # set the current texture coordinates
        glVertex3f(-width / 2, -height / 2, 0.0)
        glTexCoord2f(0, 1)
        glVertex3f(-width / 2, height / 2, 0.0)
        glTexCoord2f(1, 1)
        glVertex3f(width / 2, height / 2, 0.0)
        glTexCoord2f(1, 0)
        glVertex3f(width / 2, -height / 2, 0.0)
        glEnd()
        glPopMatrix()
        glDisable(GL_TEXTURE_2D)
