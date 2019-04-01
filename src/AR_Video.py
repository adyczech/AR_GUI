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
        self.mute = False
        self.ix = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.iy = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.init_video_texture()

        if not isinstance(self.music, type(None)):
            mixer.pre_init(44100, -16, 1, 512)
            mixer.init()
            self.sound = mixer.Sound(sound)

    def init_video_texture(self):
        ret, image = self.video.read()

        # Convert image to OpenGL texture format
        vid_image = Image.fromarray(image)
        vid_image = vid_image.tobytes("raw", "BGRX", 0, -1)

        # create video texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.gl_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.ix, self.iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, vid_image)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def start(self):
        Thread(target=self.play, args=(), daemon=True).start()

    def play(self):
        self.isPlaying = True
        nof = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.video.get(cv2.CAP_PROP_FPS)
        frame_counter = 0
        if not isinstance(self.music, type(None)):
            self.sound.play()
        tic = time.time()
        while True:
            toc = time.time()
            if not self.is_displayed and (toc - self.last_draw_time >= 0.2):
                self.sound.set_volume(0)
                self.mute = True
            elif self.is_displayed and self.mute:
                self.sound.set_volume(1)
                self.mute = False
            self.is_displayed = False

            if frame_counter == nof:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.isPlaying = False
                break

            ret, self.cur_frame = self.video.read()
            frame_counter += 1

            delay = 1/fps - (time.time() - toc)

            if delay > 0 and time.time()-tic < fps*frame_counter:
                time.sleep(delay*0.9)

    def draw_video(self, view_matrix, width, height):
        if not self.isPlaying:
            self.start()

        self.is_displayed = True
        self.last_draw_time = time.time()

        if np.size(self.cur_frame) == 1:
            return

        glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()
        glLoadMatrixd(view_matrix)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.gl_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.ix, self.iy, GL_BGR, GL_UNSIGNED_BYTE, np.flipud(self.cur_frame))

        # Draw background
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
        glDisable(GL_TEXTURE_2D)
