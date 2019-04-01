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


import math
import yaml
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from mech_ros_msgs.msg import MarkerList
from mech_ros_msgs.msg import Marker
from std_msgs.msg import Bool
from std_msgs.msg import Int8
from nav_msgs.msg import Odometry


WIDTH = 1280
HEIGHT = 720

# Font
nasa_font_12 = "src/Nasalization_Regular_12.pickle"
nasa_font_12_data = pickle.load(open(nasa_font_12, "rb"))
nasa_font_42 = "src/Nasalization_Bold_42.pickle"
nasa_font_42_data = pickle.load(open(nasa_font_42, "rb"))

# Markers
# MARKER_SIZE = 80
MARKER_SIZE = 111.8

ASTRONAUT_ID = 4
QUIZ_01_ID = 5
QUIZ_02_ID = 6
VIDEO_01_ID = 7

INVERSE_MATRIX = np.array([[1.0, 1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0, -1.0],
                           [-1.0, -1.0, -1.0, -1.0],
                           [1.0, 1.0, 1.0, 1.0]])

# Matrix for conversion from ROS frame to OpenCV in camera
R_ROS_O_camera = np.array([[  0.0,  0.0,   1.0],
 [  -1.0,   0.0,  0.0],
 [  0.0,   -1.0,   0.0]])

 # Matrix for conversion from OpenCV frame to ROS in marker
R_O_ROS_marker = np.array([[  0.0,  1.0,   0.0],
 [  0.0,   0.0,  1.0],
 [  1.0,   0.0,   0.0]])

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


### Markers in mechLAB
## Generate dictionary
my_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

## Define Aruco Detector Params
arucoParams = cv2.aruco.DetectorParameters_create()

arucoParams.adaptiveThreshConstant = 7
arucoParams.adaptiveThreshWinSizeMax = 35  # default 23
arucoParams.adaptiveThreshWinSizeMin = 3  # default 3
arucoParams.adaptiveThreshWinSizeStep = 8  # default 10

arucoParams.cornerRefinementMethod = 1
arucoParams.cornerRefinementMaxIterations = 30
arucoParams.cornerRefinementMinAccuracy = 0.01
arucoParams.cornerRefinementWinSize = 5

arucoParams.errorCorrectionRate = 0.6
arucoParams.minCornerDistanceRate = 0.05  # min distance between marker corners,
# min. distance[pix] = Perimeter*minCornerDistanceRate
arucoParams.minMarkerDistanceRate = 0.05  # min distance between corners of different markers,
# min. distance[pix] = Perimeter(smaller one)*minMarkerDistanceRate
arucoParams.minMarkerPerimeterRate = 0.1
arucoParams.maxMarkerPerimeterRate = 4.0
arucoParams.minOtsuStdDev = 5.0
arucoParams.perspectiveRemoveIgnoredMarginPerCell = 0.13
arucoParams.perspectiveRemovePixelPerCell = 8
arucoParams.polygonalApproxAccuracyRate = 0.01
arucoParams.markerBorderBits = 1
arucoParams.maxErroneousBitsInBorderRate = 0.04
arucoParams.minDistanceToBorder = 3

aspect = (WIDTH * mtx[1, 1]) / (HEIGHT * mtx[0, 0])
fovy = 2 * np.arctan(0.5 * HEIGHT / mtx[1, 1]) * 180 / np.pi

bg_dist = 45
labels = Labels()


class StreamCapture:

    def __init__(self):
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('tcpclientsrc host=mechros2.local port=8080  ! gdpdepay !  rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false', cv2.CAP_GSTREAMER)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                # self.current_frame = np.fliplr(frame)
                self.current_frame = frame
                self.time = rospy.Time.now()

    def get_current_frame(self):
        return self.current_frame, self.time


class ARGUI:

    def __init__(self):
        self.stream = StreamCapture()
        self.stream.start()

        self.win_width = WIDTH
        self.win_height = HEIGHT

        self.language = 0  # 0 - cs, 1 - eng

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
        self.voltage = 0

        self.HUD_minimap_position = np.array([0, 0])
        self.HUD_minimap_rotation = 90
        self.HUD_minimap_resolution = 39.2857142  # [px/m] 44px/1.12m

        self.cross_hair_up = False
        self.cross_hair_down = False
        self.cross_hair_right = False
        self.cross_hair_left = False

        self.fire = False
        self.queue_of_death = []  # [object, dst]

        self.tic = time.time()
        self.toc = time.time()
        self.frames = 0
        # ROS parameters
        marker_detector_topic = "/markers"
        servo_topic = "/cmd_servo"
        led_topic = "/cmd_led"
        odom_topic = "/odometry/filtered"
        volt_topic = "/volt_battery"
        self.frame_id = "front_camera_link"

        # Init subscribers and publishers
        self.marker_publisher = rospy.Publisher(marker_detector_topic, MarkerList,queue_size=10)
        self.servo_publisher = rospy.Publisher(servo_topic, Bool,queue_size=2)
        self.led_publisher = rospy.Publisher(led_topic, Bool,queue_size=2)
        odom_subscriber = rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        volt_subscriber = rospy.Subscriber(volt_topic, Int8, self.volt_cb)

        # Init Pose and velocity info
        self.x = 0
        self.y = 0
        self.velocity = 0
        self.yaw = 0

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

        # Build Font
        self.nasa_font_12 = CreateFont("src/Nasalization_Regular.ttf", 36, nasa_font_12_data)
        self.nasa_font_42 = CreateFont("src/Nasalization_Bold.ttf", 116, nasa_font_42_data)

        # Load 3D models
        # self.astronaut = OBJ('content/model/astronaut5/astronaut.obj')
        # s elf.astronaut = OBJ('content/model/med_astronaut/med_astronaut.obj')
        self.astronaut = OBJ('content/model/humanoid-robot/source/robot.obj')

        # Load HUD images
        self.HUD_bg = ARPicture('content/hud/HUD_background.png', cv2.IMREAD_UNCHANGED)
        self.HUD_cursor = ARPicture('content/hud/HUD_cursor.png', cv2.IMREAD_UNCHANGED)
        self.HUD_lights_on = ARPicture('content/hud/HUD_lights_on.png', cv2.IMREAD_UNCHANGED)
        self.HUD_lights_off = ARPicture('content/hud/HUD_lights_off.png', cv2.IMREAD_UNCHANGED)
        self.HUD_fork_up = ARPicture('content/hud/HUD_fork_up.png', cv2.IMREAD_UNCHANGED)
        self.HUD_fork_down = ARPicture('content/hud/HUD_fork_down.png', cv2.IMREAD_UNCHANGED)
        self.HUD_minimap = ARPicture('content/hud/HUD_minimap.png', cv2.IMREAD_UNCHANGED)
        self.HUD_minimap_cursor = ARPicture('content/hud/HUD_minimap_cursor.png', cv2.IMREAD_UNCHANGED)

        # Load Video
        self.video_01 = ARVideo('content/video/mars.mp4', 'content/video/mars.wav')

        # Load quizzes
        self.quiz_01 = ARQuiz(self, labels.txt[0], labels.txt[1], labels.txt[2], labels.txt[3], 2)
        self.quiz_02 = ARQuiz(self, labels.txt[4], labels.txt[5], labels.txt[6], labels.txt[7], 1)

        # Assign texture
        self.texture_background = glGenTextures(1)  # generate texture names

        self.image,_ = self.stream.get_current_frame()
        self.init_background(self.image)
        
    def glfw_main_loop(self):
        while not glfw.window_should_close(self.main_window):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.image, time = self.stream.get_current_frame()
            self.draw_background(self.image)

            ids, rvecs, tvecs, markerIds, markerCorners, my_rvecs, my_tvecs = self.find_marker(self.image)

            if len(markerCorners) > 0:
                # Publish pose estimates to ROS network
                aruco_MarkerList = MarkerList()
                aruco_MarkerList.header.stamp = time
                aruco_MarkerList.header.frame_id = self.frame_id
                if markerIds != None:
                    for i in range(markerIds.size):

                        # Calculate surface area in pixels
                        surface = cv2.contourArea(markerCorners[i], False)
                        
                        # Fill MarkerList with each marker
                        aruco_Marker = Marker()
                        aruco_Marker.id = str(markerIds[i,0])
                        aruco_Marker.surface = surface

                        # Prevedeni rodrigues vectoru na rotacni matici
                        Rmat = np.zeros(shape=(3,3))
                        cv2.Rodrigues(my_rvecs[i,0],Rmat)

                        # Convert from Opencv frame to ROS frame in camera
                        R = np.dot(R_ROS_O_camera, Rmat)

                        # Convert inverted matrix from Opencv frame to ROS frame in marker
                        R = np.dot(R, R_O_ROS_marker)

                        # Convert from Rotation matrix to Euler angles
                        Euler = self.rotationMatrixToEulerAngles(R.T) # rotace z markeru do kamery

                        # Fill Marker orientation vector
                        aruco_Marker.pose.orientation.r = Euler[0]
                        aruco_Marker.pose.orientation.p = Euler[1]
                        aruco_Marker.pose.orientation.y = Euler[2]


                        # Coordinate vector of camera position from marker in camera coordinate frame
                        aruco_Marker.pose.position.x = -my_tvecs[i,0,2]
                        aruco_Marker.pose.position.y = my_tvecs[i,0,0]
                        aruco_Marker.pose.position.z = -my_tvecs[i,0,1]

                        ## For compatibility with gazebo
                        aruco_Marker.header.stamp = time

                        # All coordinates are in marker frame
                        aruco_MarkerList.markers.append(aruco_Marker)

                    self.marker_publisher.publish(aruco_MarkerList)

            self.draw_AR(ids, rvecs, tvecs)
            self.draw_HUD()

            self.collision_evaluation()

            self.joystick()

            glfw.swap_buffers(self.main_window)
            glfw.poll_events()

            self.frames += 1

            if time.time() - self.toc >= 2:
                self.toc = time.time()
                print(self.frames/2)
                self.frames = 0

        glfw.destroy_window(self.main_window)
        glfw.terminate()

    def find_marker(self, image):
        rvecs = None
        tvecs = None
        my_rvecs = None
        my_tvecs = None
        markerIds = np.array([])
        markerCorners = np.array([])


        # image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        # corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=detector_params)
        corners, ids, rejected = cv2.aruco.detectMarkers(image, my_dictionary, parameters=arucoParams, cameraMatrix= mtx, distCoeff= dist)
        markerIds = ids
        markerCorners = corners
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
                my_rvecs = rvecs
                my_tvecs = tvecs/1000
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

        return ids, rvecs, tvecs, markerIds, markerCorners, my_rvecs, my_tvecs

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
                elif 0 == ids[i]:
                    view_matrix = self.get_view_matrix(rvecs[i], tvecs[i])
                    glColor3fv(self.font_color)
                    CreateFont.glPrint(self.nasa_font_12, 0, 0, -10, "3.6 V", view_matrix, False)
                    glColor3f(1.0, 1.0, 1.0)

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
        glBindTexture(GL_TEXTURE_2D, self.texture_background)
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
<<<<<<< 83c26254bf3ef924e2ccb8f9825410eecfed0e57
        self.HUD_minimap_position = np.array([2, -1]) * self.HUD_minimap_resolution * self.win_width / 1920
=======
        # self.
>>>>>>> Minor modify

        ###########

        self.draw_picture(self.HUD_bg, modelview, self.win_width/2, self.win_height/2, 0, self.win_width/1920, 0)
        self.draw_picture(self.HUD_minimap, modelview, round(143/1920 * self.win_width), round(151/1080 * self.win_height), 0, self.win_width / 1920, 0)
        self.draw_picture(self.HUD_minimap_cursor, modelview, round(20/1920 * self.win_width) + self.HUD_minimap_position[0],
                          round(228/1080 * self.win_height) + self.HUD_minimap_position[1], 0, self.win_width / 1920, self.HUD_minimap_rotation)

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
        CreateFont.glPrint(self.nasa_font_12, round(0.3375*self.win_width), round(0.9343*self.win_height), 0, "3.6 V", None, True)
        CreateFont.glPrint(self.nasa_font_42, round(1671/1920 * self.win_width), round(128 / 1080 * self.win_height), 0, "20", None, True)
        CreateFont.glPrint(self.nasa_font_12, round(960 / 1920 * self.win_width), round(1009 / 1080 * self.win_height), 0, "115 °", None, True)
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

    def change_language(self):
        if self.language == 0:
            self.language = 1
        else:
            self.language = 0
        labels.change_language(self.language)
        self.quiz_01.change_labels(labels.txt[0], labels.txt[1], labels.txt[2], labels.txt[3])
        self.quiz_02.change_labels(labels.txt[4], labels.txt[5], labels.txt[6], labels.txt[7])

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
            Light = Bool()
            Light.data = self.lights
            self.led_publisher.publish(Light)

        elif key == glfw.KEY_C and action == glfw.PRESS:
            self.fork = not self.fork
            Fork = Bool()
            Fork.data = self.fork
            self.servo_publisher.publish(Fork)

        elif key == glfw.KEY_V and action == glfw.PRESS:
            self.change_language()

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

    def getYaw(self, q):        
        yaw = math.atan2(2.0 * (q.z * q.w + q.x * q.y) , - 1.0 + 2.0 * (q.w * q.w + q.x * q.x))
        yaw = round(yaw*180/math.pi)
        return yaw

    def odom_cb(self,odom):
        self.x = odom.pose.pose.position.x
        self.y = odom.pose.pose.position.y
        self.yaw = self.getYaw(odom.pose.pose.orientation)
        self.velocity = round(odom.twist.twist.linear.x*100)

    def volt_cb(self,volt):
        voltage = (volt.data + 335.0)/100
        self.voltage = voltage

    def rotationMatrixToEulerAngles(self, M, cy_thresh=None):
        # cy_thresh : None or scalar, optional
        #    threshold below which to give up on straightforward arctan for
        #    estimating x rotation.  If None (default), estimate from
        #    precision of input. Source : http://www.graphicsgems.org/
        _FLOAT_EPS_4 = np.finfo(float).eps * 4.0
        if cy_thresh is None:
            try:
                cy_thresh = np.finfo(M.dtype).eps * 4
            except ValueError:
                cy_thresh = _FLOAT_EPS_4
        r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
        # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
        cy = math.sqrt(r33*r33 + r23*r23)
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = math.atan2(r21,  r22)
            y = math.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
        return [x, y, z]

    def main(self):
        # setup and run OpenGL
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 4)
        self.main_window = glfw.create_window(WIDTH, HEIGHT, "AR - GUI", None, None)
        glfw.set_window_pos(self.main_window, 200, 100)
        # glfw.set_window_aspect_ratio(self.main_window, 16, 9)
        glfw.set_window_size_callback(self.main_window, self.window_resize)

        glfw.make_context_current(self.main_window)

        glfw.set_key_callback(self.main_window, self.key_callback)

        self.init_gl()
        self.glfw_main_loop()

rospy.init_node('aruco_detect', anonymous=True)
app = ARGUI()
app.main()
