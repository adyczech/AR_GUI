from PIL import Image
import numpy as np
from OpenGL.GLU import *
from OpenGL.GL import *
import cv2
from src.AR_Picture import *


class AnswerField:
    def __init__(self):
        self.rg = ARPicture('content/quiz/answer_reg.png', cv2.IMREAD_UNCHANGED)
        self.over = ARPicture('content/quiz/answer_over.png', cv2.IMREAD_UNCHANGED)
        self.true = ARPicture('content/quiz/answer_true.png', cv2.IMREAD_UNCHANGED)
        self.false = ARPicture('content/quiz/answer_false.png', cv2.IMREAD_UNCHANGED)
        self.hover = False
        self.dead = False

class ARQuiz:

    def __init__(self, parent, question, answer1, answer2, answer3, correct_answer):
        self.parent = parent

        self.correct_answer = correct_answer
        self.answered = False
        self.answered_correctly = False
        self.choice = 0

        self.question = ARPicture(question, cv2.IMREAD_UNCHANGED)
        self.answer_1 = ARPicture(answer1, cv2.IMREAD_UNCHANGED)
        self.answer_2 = ARPicture(answer2, cv2.IMREAD_UNCHANGED)
        self.answer_3 = ARPicture(answer3, cv2.IMREAD_UNCHANGED)

        self.answer_field_1 = AnswerField()
        self.answer_field_2 = AnswerField()
        self.answer_field_3 = AnswerField()

    def draw_quiz(self, view_matrix):
        self.question.set_modelview_matrix(view_matrix, 0, self.answer_field_1.rg.iy + self.question.iy/2, 0)
        self.question.draw()

        self.draw_picture(self.answer_field_1, view_matrix, 0, self.answer_field_1.rg.iy / 2, 0, hitable=True, index=1)
        self.draw_picture(self.answer_field_2, view_matrix, 0, -self.answer_field_2.rg.iy / 2, 0, hitable=True, index=2)
        self.draw_picture(self.answer_field_3, view_matrix, 0, -1.5 * self.answer_field_3.rg.iy, 0, hitable=True, index=3)

        self.answer_1.set_modelview_matrix(view_matrix, 0, self.answer_field_1.rg.iy / 2, 3)
        self.answer_1.draw()
        self.answer_2.set_modelview_matrix(view_matrix, 0, -self.answer_field_1.rg.iy / 2, 3)
        self.answer_2.draw()
        self.answer_3.set_modelview_matrix(view_matrix, 0, -1.5 * self.answer_field_1.rg.iy, 3)
        self.answer_3.draw()

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

    def draw_picture(self, field, view_matrix, x, y, z, scale=1, rz=0, hitable=False, index=0):
        modelview_mat = field.rg.set_modelview_matrix(view_matrix, x, y, z, scale, rz)

        if hitable:
            if self.choice == 0:
                model_mat = np.transpose(modelview_mat)
                hit, dst = self.collision_detection(field.rg.BB, model_mat, scale * scale)
                if hit:
                    self.parent.queue_of_death.append([field, dst])

            if not field.dead:
                if field.hover and self.choice == 0:
                    picture = field.over
                    field.hover = False
                elif not field.hover and self.choice == 0:
                    picture = field.rg
                elif self.choice != 0 and index == self.correct_answer:
                    picture = field.true
                else:
                    picture = field.rg
            else:
                if self.choice == 0:
                    self.choice = index

                if index == self.correct_answer:
                    picture = field.true
                elif index == self.choice and self.choice != self.correct_answer:
                    picture = field.false
                elif index != self.choice and index != self.correct_answer:
                    picture = field.rg

        picture.draw(modelview_mat)

        # if hitable:
        #     if self.choice == 0:
        #         model_mat = np.transpose(modelview_mat)
        #         hit, dst = self.collision_detection(picture.rg.BB, model_mat, scale * scale)
        #         if hit:
        #             if self.parent.fire:
        #                 self.parent.fire = False
        #                 self.choice = index
        #                 if index == self.correct_answer:
        #                     picture = picture.true
        #                     self.answered_correctly = True
        #                 else:
        #                     picture = picture.false
        #             else:
        #                 picture = picture.rg
        #         else:
        #             picture = picture.over
        #     else:
        #         if index == self.correct_answer:
        #             picture = picture.true
        #         elif index == self.choice and self.choice != self.correct_answer:
        #             picture = picture.false
        #         elif index != self.choice and index != self.correct_answer:
        #             picture = picture.rg
        # else:
        #     picture = picture
        #
        # if picture.blend:
        #     glEnable(GL_BLEND)
        #     glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #
        # glEnable(GL_TEXTURE_2D)
        # glCallList(picture.gl_list)
        # glDisable(GL_TEXTURE_2D)
        #
        # if picture.blend:
        #     glDisable(GL_BLEND)
        #
        # glPopMatrix()
