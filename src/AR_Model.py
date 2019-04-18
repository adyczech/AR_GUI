from src.objloader_V2 import *
import numpy as np
from OpenGL.GLU import *


class ARModel:

    def __init__(self, path, hitable=False, parent=None):
        self.mesh = OBJ(path)
        self.hitable = hitable
        self.parent = parent
        self.dead = False
        self.hover = False

    def draw_model(self, view_matrix, x, y, z, scale, rx, ry, rz):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glLoadMatrixd(view_matrix)

        glTranslatef(x, y, z)
        glScale(scale, scale, scale)  # Scale model

        # Rotate model
        glRotatef(rx, 1, 0, 0)
        glRotatef(ry, 0, 1, 0)
        glRotatef(rz, 0, 0, 1)

        modelview_mat = glGetFloat(GL_MODELVIEW_MATRIX)

        if self.hitable:
                model_mat = np.transpose(modelview_mat)
                hit, dst = self.collision_detection(self.mesh.BB, model_mat, scale * scale)
                if hit:
                    self.parent.queue_of_death.append([self, dst])

                if not self.dead:
                    glEnable(GL_TEXTURE_2D)
                    glCallList(self.mesh.gl_list)
                    glDisable(GL_TEXTURE_2D)
                    glColor3f(1, 1, 1)
        else:
            glEnable(GL_TEXTURE_2D)
            glCallList(self.mesh.gl_list)
            glDisable(GL_TEXTURE_2D)
            glColor3f(1, 1, 1)
            
    def collision_detection(self, BB, model_mat, scale):
        ray_orig = np.array([[0], [0], [0]])

        win_x = self.parent.cross_hair_x
        win_y = self.parent.cross_hair_y

        projection = glGetDouble(GL_PROJECTION_MATRIX)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        modelview = glGetDouble(GL_MODELVIEW_MATRIX)
        glPopMatrix()
        viewport = glGetInteger(GL_VIEWPORT)

        start_x, start_y, start_z = gluUnProject(win_x, win_y, 0.0, modelview, projection, viewport)
        end_x, end_y, end_z = gluUnProject(win_x, win_y, 1.0, modelview, projection, viewport)

        ray_start = np.array([[start_x], [start_y], [start_z]])
        ray_end = np.array([[end_x], [end_y], [end_z]])

        ray_dir = ray_end - ray_start
        ray_dir = np.transpose(ray_dir/np.linalg.norm(ray_dir))[0]

        tmin = 0
        tmax = float("inf")

        OBBposition_worldspace = np.array([[model_mat[0][3]], [model_mat[1][3]], [model_mat[2][3]]])
        delta = np.transpose(OBBposition_worldspace - ray_orig)[0]

        for i in range(0, 3):
            axis = np.array([model_mat[0][i], model_mat[1][i], model_mat[2][i]])

            e = np.dot(axis, delta)
            f = np.dot(ray_dir, axis)

            t1 = (e + BB[0][i] * scale) / f
            t2 = (e + BB[1][i] * scale) / f

            if t1 > t2:
                t1, t2 = t2, t1

            if t2 < tmax:
                tmax = t2

            if t1 > tmin:
                tmin = t1

        hit = tmax > tmin
        dst = min(t1, t2)
        return hit, dst