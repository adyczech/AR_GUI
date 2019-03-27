from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import glm
import cv2
from PIL import Image
import numpy as np
from threading import Thread
import threading
import time
from operator import itemgetter
import pickle
from pygame import mixer
from src.objloader_V2 import *
from src.Labels import *
from src.AR_Picture import *
from src.AR_Quiz import *
from src.AR_Video import *
from src.CreateFont import *

WIDTH = 1280
HEIGHT = 720

# Font
nasa_font_12 = "src/Nasalization_Regular_12.pickle"
nasa_font_12_data = pickle.load(open(nasa_font_12, "rb"))
nasa_font_42 = "src/Nasalization_Bold_42.pickle"
nasa_font_42_data = pickle.load(open(nasa_font_42, "rb"))

# Markers
# MARKER_SIZE = 80
MARKER_SIZE = 140

ASTRONAUT_ID = 4
QUIZ_01_ID = 5
QUIZ_02_ID = 6
VIDEO_01_ID = 7

INVERSE_MATRIX = np.array([[1.0, 1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0, -1.0],
                           [-1.0, -1.0, -1.0, -1.0],
                           [1.0, 1.0, 1.0, 1.0]])

# Load camera matrix and distortion coefficients
with np.load('src/camCal_1280x720_MS.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# Marker parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector_params = cv2.aruco.DetectorParameters_create()
detector_params.cornerRefinementMethod = 1
detector_params.adaptiveThreshWinSizeStep = 5
detector_params.minMarkerPerimeterRate = 0.04
detector_params.cornerRefinementMaxIterations = 30

aspect = (WIDTH * mtx[1, 1]) / (HEIGHT * mtx[0, 0])
fovy = 2 * np.arctan(0.5 * HEIGHT / mtx[1, 1]) * 180 / np.pi

bg_dist = 45
labels = Labels()


class StreamCapture:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        # self.cap.set(cv2.CAP_PROP_SETTINGS, 1)
        if not self.cap.isOpened():
            print("Cannot open stream")
            exit()
        else:
            print("Stream OPENED")

    def start(self):
        Thread(target=self.update_frame, args=(), daemon=True).start()

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = np.fliplr(frame)

    def get_current_frame(self):
        return self.current_frame


class ARGUI:

    def __init__(self):
        self.stream = StreamCapture()
        self.stream.start()

        self.win_width = WIDTH
        self.win_height = HEIGHT

        self.cross_hair_x = self.win_width / 2
        self.cross_hair_y = self.win_height / 2

        self.font_color = [0.933, 0.937, 0.941]

        # Marker probability
        self.last_ids_mtx = np.zeros((50, 10), dtype=np.int8)
        self.prob_vect = np.zeros(50, dtype=np.float16)
        self.last_corners = np.zeros((50, 1, 4, 2), dtype=np.float32)
        self.corners_avg_buffer = np.zeros((50, 3, 4, 2), dtype=np.float32)

        self.lights = False
        self.fork = False

        self.HUD_minimap_position = [0, 0]

        self.cross_hair_up = False
        self.cross_hair_down = False
        self.cross_hair_right = False
        self.cross_hair_left = False

        self.fire = False
        self.queue_of_death = []  # [object, dst]

    def init_gl(self):
        print("init")
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)  # specify the value used for depth buffer comparisons
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_MULTISAMPLE)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        glMatrixMode(GL_PROJECTION)  # specify which matrix is the current matrix
        glLoadIdentity()
        gluPerspective(fovy, aspect, 0.1,
                       500.0 * MARKER_SIZE)  # field of view, aspect ration, distance to near clipping plane, distance to far clipping plane

        # Load 3D models
        # self.astronaut = OBJ('content/model/astronaut5/astronaut.obj')
        # self.astronaut = OBJ('content/model/med_astronaut/med_astronaut.obj')
        self.astronaut = OBJ('content/model/humanoid-robot/source/robot.obj')

        # Load HUD images
        self.HUD_bg = ARPicture('content/hud/HUD_background.png', cv2.IMREAD_UNCHANGED)
        self.HUD_cursor = ARPicture('content/hud/HUD_cursor.png', cv2.IMREAD_UNCHANGED)
        self.HUD_lights_on = ARPicture('content/hud/HUD_lights_on.png', cv2.IMREAD_UNCHANGED)
        self.HUD_lights_off = ARPicture('content/hud/HUD_lights_off.png', cv2.IMREAD_UNCHANGED)
        self.HUD_fork_up = ARPicture('content/hud/HUD_fork_up.png', cv2.IMREAD_UNCHANGED)
        self.HUD_fork_down = ARPicture('content/hud/HUD_fork_down.png', cv2.IMREAD_UNCHANGED)
        self.HUD_minimap = ARPicture('content/hud/HUD_minimap.png', cv2.IMREAD_UNCHANGED)
        self.HUD_minimap_cursor = ARPicture('content/hud/HUD_minimap.png', cv2.IMREAD_UNCHANGED)

        # Load Video
        self.video_01 = ARVideo('content/video/mars.mp4', 'content/video/mars.wav')

        # Load quizzes
        self.quiz_01 = ARQuiz(self, 'content/quiz/1_0.png', 'content/quiz/1_a.png', 'content/quiz/1_b.png', 'content/quiz/1_c.png', 2)
        self.quiz_02 = ARQuiz(self, 'content/quiz/2_0.png', 'content/quiz/2_a.png', 'content/quiz/2_b.png',
                              'content/quiz/2_c.png', 1)

        # Build Font
        self.nasa_font_12 = CreateFont("src/Nasalization_Regular.ttf", 34, nasa_font_12_data)  # 34
        self.nasa_font_42 = CreateFont("src/Nasalization_Bold.ttf", 116, nasa_font_42_data)

        # Assign texture
        self.texture_background = glGenTextures(1)  # generate texture names

        self.image = self.stream.get_current_frame()
        self.init_background(self.image)

    def glfw_main_loop(self):
        while not glfw.window_should_close(self.main_window):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.image = self.stream.get_current_frame()
            self.draw_background(self.image)

            ids, rvecs, tvecs = self.find_marker(self.image)

            self.draw_AR(ids, rvecs, tvecs)
            self.draw_HUD()

            self.collision_evaluation()

            self.joystick()

            glfw.swap_buffers(self.main_window)
            glfw.poll_events()

        glfw.destroy_window(self.main_window)
        glfw.terminate()

    def find_marker(self, image):
        rvecs = None
        tvecs = None

        # image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=detector_params)
        # corners = np.dot(corners, 2)

        # <editor-fold desc = "Probability" >
        if not isinstance(ids, type(None)):
            self.last_ids_mtx[ids, 0] = 1
            self.last_corners[ids, 0, :, :] = np.float32(corners)

        self.prob_vect = np.sum(self.last_ids_mtx, 1) / np.size(self.last_ids_mtx, 1)
        self.last_ids_mtx = np.pad(self.last_ids_mtx, ((0, 0), (1, 0)), mode='constant')[:, :-1]

        if np.any(self.prob_vect > 0.3) & np.any(self.prob_vect < 1):
            prob_idx = np.array(np.where(np.logical_and(self.prob_vect > 0.3, self.prob_vect < 1)))[0]
            # print("P1 " + str(prob_idx))
            if not isinstance(ids, type(None)):
                to_delete = []
                # print(np.size(prob_idx, axis=1))
                for i in range(0, np.size(prob_idx)):
                    if prob_idx[i] in ids:
                        to_delete.append(i)
                prob_idx = np.delete(prob_idx, to_delete)
            # print("P2 " + str(prob_idx))
            if isinstance(ids, type(None)):
                ids = np.array([[0]])
                erase = True
            else:
                erase = False
            # print("Mezi1 " + str(ids))
            for i in range(0, np.size(prob_idx)):
                ids = np.append(ids, [[prob_idx[i]]], axis=0)
                # print(np.float32([self.last_corners[prob_idx[i], 0, :, :]]))
                corners.append(np.float32([self.last_corners[prob_idx[i], 0, :, :]]))
            # print("Mezi1 " + str(ids))
            if erase:
                ids = np.delete(ids, ids[0], axis=0)
            if np.size(ids) == 0:
                ids = None
        # </editor-fold >

        # Averaging
        if not isinstance(ids, type(None)):
            self.corners_avg_buffer[ids, 0, :, :] = corners
            corners_avg = np.sum(self.corners_avg_buffer, 1) / 3
            valid_avg = np.sum(np.prod(self.corners_avg_buffer, 1), (1, 2))

            for i in range(0, np.size(ids, 0)):
                if valid_avg[ids[i]] != 0:
                    corners[i] = corners_avg[ids[i][0], :, :]

        self.corners_avg_buffer = np.pad(self.corners_avg_buffer, ((0, 0), (1, 0), (0, 0), (0, 0)), mode='constant')[:,
                                  :-1, :, :]

        if not isinstance(ids, type(None)):

            if not isinstance(ids, type(None)):
                rvecs, tvecs, origin = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
                rvecs = rvecs[:, 0, :]
                tvecs = tvecs[:, 0, :]
                ids = ids[:, 0]
            else:
                ids = np.array([0])
                tvecs = np.array([[0, 0, 0]])
                rvecs = np.array([[0, 0, 0]])

                ids = np.delete(ids, ids[0], axis=0)
                tvecs = np.delete(tvecs, tvecs[0], axis=0)
                rvecs = np.delete(rvecs, rvecs[0], axis=0)

            # Sorting according to z distanced
            inds = np.flipud(np.argsort(tvecs[:, 2], 0))
            tvecs = tvecs[inds]
            rvecs = rvecs[inds]
            ids = ids[inds]

        return ids, rvecs, tvecs

    def draw_AR(self, ids, rvecs, tvecs):
        if not isinstance(ids, type(None)):
            # a = time.time()
            for i in range(0, np.size(ids)):
                if ASTRONAUT_ID == ids[i]:
                    view_matrix = self.get_view_matrix(rvecs[i], tvecs[i])
                    self.draw_model(self.astronaut, view_matrix, 50, 0, 0, 0, 0, 0, 0)
                elif QUIZ_01_ID == ids[i]:
                    view_matrix = self.get_view_matrix(rvecs[i], tvecs[i])
                    self.quiz_01.draw_quiz(view_matrix)
                elif QUIZ_02_ID == ids[i]:
                    view_matrix = self.get_view_matrix(rvecs[i], tvecs[i])
                    self.quiz_02.draw_quiz(view_matrix)
                elif VIDEO_01_ID == ids[i]:
                    view_matrix = self.get_view_matrix(rvecs[i], tvecs[i])
                    self.video_01.draw_video(view_matrix, 640, 480)

    def get_view_matrix(self, rvecs, tvecs):
        model_rvecs = rvecs
        model_tvecs = tvecs
        rmtx = cv2.Rodrigues(model_rvecs)[0]

        view_matrix = np.array([[rmtx[0, 0], rmtx[0, 1], rmtx[0, 2], model_tvecs[0]],
                                [rmtx[1, 0], rmtx[1, 1], rmtx[1, 2], model_tvecs[1]],
                                [rmtx[2, 0], rmtx[2, 1], rmtx[2, 2], model_tvecs[2] * 1],
                                [0.0, 0.0, 0.0, 1.0]])

        view_matrix = np.transpose(view_matrix * INVERSE_MATRIX)
        return view_matrix

    def draw_model(self, model, view_matrix, scale, x, y, z, rx, ry, rz):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glPushMatrix()
        glLoadMatrixd(view_matrix)

        glTranslatef(x, y, z)
        glScale(scale, scale, scale)  # Scale model

        # Rotate model
        glRotatef(rx, 1, 0, 0)
        glRotatef(ry, 0, 1, 0)
        glRotatef(rz, 0, 0, 1)

        glEnable(GL_TEXTURE_2D)
        glCallList(model.gl_list)
        glDisable(GL_TEXTURE_2D)
        glColor3f(1, 1, 1)
        glPopMatrix()

    def draw_picture(self, picture, view_matrix, x, y, z, scale=1.0, rz=0):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if picture.blend:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_TEXTURE_2D)

        glPushMatrix()
        glLoadMatrixd(view_matrix)
        glTranslatef(x, y, z)
        glScale(scale, scale, scale)
        glRotatef(rz, 0, 0, 1)

        glCallList(picture.gl_list)

        if picture.blend:
            glDisable(GL_BLEND)

        glDisable(GL_TEXTURE_2D)

        glPopMatrix()

    def draw_background(self, image):
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D,
                      self.texture_background)  # bind a named texture to a texturing target (target, texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, np.flipud(image))

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Move background
        glTranslatef(0.0, 0.0, -bg_dist * MARKER_SIZE)

        fn = 0.83  # 0.83 camCal_1280x720_MS
        # fn=1.2
        # fn = 1.85

        # Draw background
        glBegin(GL_QUADS)
        glTexCoord2f(1, 1)  # set the current texture coordinates
        glVertex3f(0.8 * bg_dist * MARKER_SIZE * fn, 0.45 * bg_dist * MARKER_SIZE * fn, 0.0)
        glTexCoord2f(1, 0)
        glVertex3f(0.8 * bg_dist * MARKER_SIZE * fn, -0.45 * bg_dist * MARKER_SIZE * fn, 0.0)
        glTexCoord2f(0, 0)
        glVertex3f(-0.8 * bg_dist * MARKER_SIZE * fn, -0.45 * bg_dist * MARKER_SIZE * fn, 0.0)
        glTexCoord2f(0, 1)
        glVertex3f(-0.8 * bg_dist * MARKER_SIZE * fn, 0.45 * bg_dist * MARKER_SIZE * fn, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def init_background(self, image):
        # Convert image to OpenGL texture format
        bg_image = Image.fromarray(image)
        ix = bg_image.size[0]
        iy = bg_image.size[1]
        bg_image = bg_image.tobytes("raw", "BGRX", 0, -1)

        # create background texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, bg_image)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

    def draw_HUD(self):
        self.ortho_view()  # Set ortho view
        glDisable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        modelview = glGetDouble(GL_MODELVIEW_MATRIX)
        glPopMatrix()

        # UPRAVIT #
        self.

        ###########


        self.draw_picture(self.HUD_bg, modelview, self.win_width/2, self.win_height/2, 0, self.win_width/1920, 0)
        self.draw_picture(self.HUD_minimap, modelview, round(143/1920 * self.win_width), round(151/1080 * self.win_height), 0, self.win_width / 1920, 0)
        self.draw_picture(self.HUD_minimap_cursor, modelview, round(143/1920 * self.win_width), round(151/1080 * self.win_height), 0, self.win_width / 1920, 0)

        if self.lights:
            self.draw_picture(self.HUD_lights_on, modelview, round(1230/1920 * self.win_width), round(1030/1080 * self.win_height), 0, self.win_width / 1920, 0)
        else:
            self.draw_picture(self.HUD_lights_off, modelview, round(1230/1920 * self.win_width), round(1030/1080 * self.win_height), 0, self.win_width / 1920, 0)

        if self.fork:
            self.draw_picture(self.HUD_fork_up, modelview, round(535/1920 * self.win_width), round(1030/1080 * self.win_height), 0, self.win_width / 1920, 0)
        else:
            self.draw_picture(self.HUD_fork_down, modelview, round(535/1920 * self.win_width), round(1030/1080 * self.win_height), 0, self.win_width / 1920, 0)

        # Draw HUD texts
        glColor3fv(self.font_color)
        CreateFont.glPrint(self.nasa_font_12, round(0.3375*self.win_width), round(0.9343*self.win_height), "3.6 V")
        CreateFont.glPrint(self.nasa_font_42, round(1671/1920 * self.win_width), round(128 / 1080 * self.win_height), "20")
        CreateFont.glPrint(self.nasa_font_12, round(960 / 1920 * self.win_width), round(1009 / 1080 * self.win_height), "115 °")
        glColor3f(1.0, 1.0, 1.0)

        # Draw cursor
        self.draw_picture(self.HUD_cursor, modelview, self.cross_hair_x, self.cross_hair_y, 0, self.win_width / 1920, 0)

        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        self.perspective_view()  # Restore perspective view

    def collision_evaluation(self):
        if len(self.queue_of_death) > 0:
            self.queue_of_death = sorted(self.queue_of_death, key=itemgetter(1))
            if not self.queue_of_death[0][0].hover:
                self.queue_of_death[0][0].hover = True
            if not self.queue_of_death[0][0].dead and self.fire:
                self.queue_of_death[0][0].dead = True
            self.queue_of_death = []
        self.fire = False

    def ortho_view(self):
        glMatrixMode(GL_PROJECTION)  # Select Projection
        glPushMatrix()  # Push The Matrix
        glLoadIdentity()  # Reset The Matrix
        glOrtho(0, self.win_width, 0, self.win_height, 0, 1)  # Set ortho projection
        glMatrixMode(GL_MODELVIEW)  # Select Modelview Matrix
        glPushMatrix()  # Push The Matrix
        glLoadIdentity()

    def perspective_view(self):
        glMatrixMode(GL_PROJECTION)  # Select Projection
        glPopMatrix()  # Pop The Matrix
        glMatrixMode(GL_MODELVIEW)  # Select Modelview
        glPopMatrix()  # Pop The Matrix

    def window_resize(self, window, width, height):
        if width != 0 or height != 0:
            self.win_width = width
            self.win_height = height
            aspect = (width * mtx[1, 1]) / (height * mtx[0, 0])
            glViewport(0, 0, width, height)
            gluPerspective(fovy, aspect, 0.1, 500.0 * MARKER_SIZE)

    def draw_cube_model(self, cube_ids, cube_size, model, id, rvec, tvec, scale, x, y, z, rx, ry, rz):
        if cube_ids[0] == id:  # 6
            view_matrix = self.get_view_matrix(rvec, tvec)
            self.draw_model(model, view_matrix, scale, x, y, -cube_size / 2 + z, 0, 0, 0)
        elif cube_ids[1] == id:  # 7
            view_matrix = self.get_view_matrix(rvec, tvec)
            self.draw_model(model, view_matrix, scale, x, y, -cube_size / 2, 0, 90, 0)
        elif cube_ids[2] == id:  # 8
            view_matrix = self.get_view_matrix(rvec, tvec)
            self.draw_model(model, view_matrix, scale, x, y, -cube_size / 2, 0, 180, 0)
        elif cube_ids[3] == id:  # 9
            view_matrix = self.get_view_matrix(rvec, tvec)
            self.draw_model(model, view_matrix, scale, x, y, -cube_size / 2, 0, 270, 0)
        elif cube_ids[4] == id:  # 10
            view_matrix = self.get_view_matrix(rvec, tvec)
            self.draw_model(model, view_matrix, scale, x, -z, y - cube_size / 2, 90, 0, 0)
        elif cube_ids[5] == id:  # 11
            view_matrix = self.get_view_matrix(rvec, tvec)
            self.draw_model(model, view_matrix, scale, x, z, -y - cube_size / 2, -90, 0, 0)

    def joystick(self):
        # y, x, _, a, b = glfw.get_joystick_axes(glfw.JOYSTICK_1)[0][0:5]
        # # print([x,y,a,b])
        #
        # if x < -0.6 and self.cross_hair_x < self.win_width:
        #     self.cross_hair_x += 20
        # elif x > 0.6 and self.cross_hair_x > 0:
        #     self.cross_hair_x -= 20
        #
        # if y < -0.6 and self.cross_hair_y > 0:
        #     self.cross_hair_y -= 20
        # elif y > 0.6 and self.cross_hair_y < self.win_height:
        #     self.cross_hair_y += 20
        if self.cross_hair_up and self.cross_hair_y < self.win_height:
            self.cross_hair_y += 20
        if self.cross_hair_down and self.cross_hair_y > 0:
            self.cross_hair_y -= 20
        if self.cross_hair_right and self.cross_hair_x < self.win_width:
            self.cross_hair_x += 20
        if self.cross_hair_left and self.cross_hair_x > 0:
            self.cross_hair_x -= 20

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_X and action == glfw.PRESS:
            self.lights = not self.lights

        elif key == glfw.KEY_C and action == glfw.PRESS:
            self.fork = not self.fork

        elif key == glfw.KEY_K and action == glfw.PRESS:
            self.fire = True

        elif key == glfw.KEY_U and action == glfw.PRESS:
            self.cross_hair_up = True

        elif key == glfw.KEY_P and action == glfw.PRESS:
            self.cross_hair_down = True

        elif key == glfw.KEY_O and action == glfw.PRESS:
            self.cross_hair_right = True

        elif key == glfw.KEY_I and action == glfw.PRESS:
            self.cross_hair_left = True

        elif key == glfw.KEY_U and action == glfw.RELEASE:
            self.cross_hair_up = False

        elif key == glfw.KEY_P and action == glfw.RELEASE:
            self.cross_hair_down = False

        elif key == glfw.KEY_O and action == glfw.RELEASE:
            self.cross_hair_right = False

        elif key == glfw.KEY_I and action == glfw.RELEASE:
            self.cross_hair_left = False

    def main(self):
        # setup and run OpenGL
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 4)
        self.main_window = glfw.create_window(WIDTH, HEIGHT, "AR - GUI", None, None)
        glfw.set_window_pos(self.main_window, 200, 100)
        glfw.set_window_aspect_ratio(self.main_window, 16, 9)
        glfw.set_window_size_callback(self.main_window, self.window_resize)

        glfw.make_context_current(self.main_window)

        glfw.set_key_callback(self.main_window, self.key_callback)

        self.init_gl()
        self.glfw_main_loop()


app = ARGUI()
app.main()
