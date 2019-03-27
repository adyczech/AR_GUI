from PIL import Image
import numpy as np
from OpenGL.GLU import *
from OpenGL.GL import *
import cv2


class ARPicture():

    def __init__(self, path, mode=cv2.IMREAD_COLOR):
        img = cv2.imread(path, mode)
        self.gl_list = glGenLists(1)
        self.gl_texture = glGenTextures(1)
        self.blend = False
        self.modelview_mat = None

        # convert image to OpenGL texture format
        picture = Image.fromarray(img)
        self.ix = picture.size[0]
        self.iy = picture.size[1]

        self.BB = [[-self.ix / 2, -self.iy / 2, -1],
                   [self.ix / 2, self.iy / 2, 1]]  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]

        if np.shape(picture)[2] == 4:
            picture = picture.tobytes("raw", "BGRA", 0, -1)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.blend = True
        else:
            picture = picture.tobytes("raw", "BGRX", 0, -1)

        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D,
                      self.gl_texture)  # bind a named texture to a texturing target (target, texture)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.ix, self.iy, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     picture)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)  # set the current texture coordinates
        glVertex3f(-self.ix / 2, -self.iy / 2, 0.0)
        glTexCoord2f(0, 1)
        glVertex3f(-self.ix / 2, self.iy / 2, 0.0)
        glTexCoord2f(1, 1)
        glVertex3f(self.ix / 2, self.iy / 2, 0.0)
        glTexCoord2f(1, 0)
        glVertex3f(self.ix / 2, -self.iy / 2, 0.0)
        glEnd()
        glEndList()

        del img, picture

    def set_modelview_matrix(self, view_matrix, x, y, z, scale=1, rz=0):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glLoadMatrixd(view_matrix)
        glTranslatef(x, y, z)
        glScale(scale, scale, scale)
        glRotatef(rz, 0, 0, 1)

        self.modelview_mat = glGetFloat(GL_MODELVIEW_MATRIX)
        return self.modelview_mat

    def draw(self, *args):
        for arg in args:
            self.modelview_mat = arg

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixd(self.modelview_mat)

        if self.blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_TEXTURE_2D)
        glCallList(self.gl_list)
        glDisable(GL_TEXTURE_2D)

        if self.blend:
            glDisable(GL_BLEND)
