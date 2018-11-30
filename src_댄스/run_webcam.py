# icon은 'Flaticon'에서 가져왔으며,  CC 3.0 BY.의 저작권 보호를 받습니다.
# 복제, 배포, 전시, 공연 및 공중송신과 변형, 2차적 저작물 작성 및 영리목적 이용가능합니다.
# 사용시마다 Flaticon의 저작권 표시를 권장합니다.

import pyrealsense2 as rs
# pyrealsense sdk 2.0ver import
import math
import mysql_update as mysql
from socket import *

import tensorflow as tf
import argparse
import time
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import importlib
import os
from config import FLAGS
from utils import utils, cpm_utils, tracking_module
import matplotlib.pyplot as plt             # Convexhull import
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

HOST = '127.0.0.1'
PORT = 56789
BUFSIZ = 1024
ADDR = (HOST, PORT)
tcpCliSock = socket(AF_INET, SOCK_STREAM)

aaa = 1
bbb = 1
fps_time = 0
kkk = 0
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# pipeline is that import realsense 3 modules.
# config streaming values set your camera configure.
# enable_stream(camera_module, resolution_w,resolution_h,pixel_format,frame)
# frame was fixed values.

joint_pixelX = [[1 for i in range(19)] for j in range(4)]  # record coordinates-X
joint_pixelY = [[1 for i in range(19)] for j in range(4)]  # record coordinates-Y
joint_dist = [[0 for i in range(19)] for j in range(4)] # distance array values setup
# joint_pointX = [[1 for i in range(19)] for j in range(4)] # for return value of axisX
# joint_pointY = [[1 for i in range(19)] for j in range(4)] # for return value of axisY
frame_add = 0
angle_filter = [[[0 for i in range(10)] for j in range(10)] for l in range(6)]
angle_filter_Rhand = [0 for i in range(21)]
angle_filter_Lhand = [0 for i in range(21)]
coordinates=[[0 for i in range(19)]]
Joint_Record = False
Hand_connect = False
hand_info = False
Unity_control = True
option_push = False
Set_People = False
hand_joint_dist_left = np.zeros(21)
hand_joint_dist_right = np.zeros(21)
local_img_left = np.zeros(shape=(240,240,3))
local_img_right = np.zeros(shape=(240,240,3))
handPoint3dX=[0 for i in range(21)]
handPoint3dY=[0 for i in range(21)]
handPoint3dZ=[0 for i in range(21)]
point3dX = [[1 for i in range(19)] for j in range(4)] # for return value of axisX
point3dY = [[1 for i in range(19)] for j in range(4)] # for return value of axisY
point3dZ = [[0 for i in range(19)] for j in range(4)] # distance array values setup
last_jointX = 0
last_jointY = 0
Axis_array = [[0 for i in range(3)] for j in range(19)]
# pts = np.zeros((1000,3))
Img_left = np.zeros(shape=(400,400,3))
Img_right  = np.zeros(shape=(400,400,3))

x_temp =np.zeros(shape=(6,19),dtype=np.float)
y_temp =np.zeros(shape=(6,19),dtype=np.float)
z_temp =np.zeros(shape=(6,19),dtype=np.float)
angleY = 9
angleZ = 9


def normalize_and_centralize_img(img):
    if FLAGS.color_channel == 'GRAY':
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).reshape((FLAGS.input_size, FLAGS.input_size, 1))

    if FLAGS.normalize_img:
        test_img_input = img / 256.0 - 0.5
        test_img_input = np.expand_dims(test_img_input, axis=0)
    else:
        test_img_input = img - 128.0
        test_img_input = np.expand_dims(test_img_input, axis=0)
    return test_img_input

def visualize_result(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img, center_axis):
    demo_stage_heatmaps = []
    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
    last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))
    correct_and_draw_hand(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img, center_axis)
    return crop_img

def visualize_result_flip(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img, center_axis):
    demo_stage_heatmaps = []
    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
    last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))
    correct_and_draw_hand_flip(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img, center_axis)
    return crop_img




def correct_and_draw_hand(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img, center_axis):
    global joint_detections_right
    global local_joint_right
    global image_rgb
    joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
    local_joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
    mean_response_val = 0.0
    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            #joint_coord[0] and joint_coord[1] difference ??? by hc
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
            local_joint_coord_set[joint_num, :] = correct_coord
            local_joint_right[joint_num, :] = correct_coord


            # Resize back
            correct_coord /= crop_full_scale

            # Substract padding border
            correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
            correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
            if center_axis :
                correct_coord[0] += int(tracker.bbox[0]-175+center_axis[1])
                correct_coord[1] += int(tracker.bbox[2]-95+center_axis[0])
                if joint_num == 0:
                    correct_coord[0] = int(center_axis[1])
                    correct_coord[1] = int(center_axis[0])
            joint_coord_set[joint_num, :] = correct_coord
    if tracker.loss_track:
        #print("loss_track..")
        #joint_coords = FLAGS.default_hand
        draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
        joint_detections_right = joint_coord_set
    else:
        draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
        if center_axis:
            draw_hand(image_rgb, joint_coord_set, tracker.loss_track)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
            image_bg = image_rgb[396:460, 20:84]
            image_bg = cv2.add(hand_icon_right, image_bg)
            image_rgb[396:460, 20:84] = image_bg
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
            # 이때 손 모양 아이콘 출력.
        joint_detections_right = joint_coord_set

    if mean_response_val >= 1:
        tracker.loss_track = False
    else:
        tracker.loss_track = True



def correct_and_draw_hand_flip(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img,center_axis):
    global joint_detections_left
    global local_joint_left
    global image_rgb
    joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
    local_joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))

    mean_response_val = 0.0
    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            #joint_coord[0] and joint_coord[1] difference ??? by hc
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
            local_joint_coord_set[joint_num, :] = correct_coord
            local_joint_left[joint_num, :] = correct_coord
            # Resize back
            correct_coord /= crop_full_scale

            # Substract padding border
            correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
            correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
            if center_axis :
                correct_coord[0] += int(tracker.bbox[0] - 175 + center_axis[1])
                correct_coord[1] = int(640-((tracker.bbox[2] - 95 + center_axis[0])+correct_coord[1]))
                if joint_num == 0:
                    correct_coord[0] = int(center_axis[1])
                    correct_coord[1] = int(640-center_axis[0])
            joint_coord_set[joint_num, :] = correct_coord
    if tracker.loss_track:
        #print("loss_track..")
        #joint_coords = FLAGS.default_hand
        draw_hand_flip(crop_img, local_joint_coord_set, tracker.loss_track)
        joint_detections_left = joint_coord_set
    else:
        draw_hand_flip(crop_img, local_joint_coord_set, tracker.loss_track)
        if center_axis:
            draw_hand_flip(image_rgb, joint_coord_set, tracker.loss_track)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
            image_bg = image_rgb[396:460, 556:620]
            image_bg = cv2.add(hand_icon_left, image_bg)
            image_rgb[396:460, 556:620] = image_bg
            # Img_left = cv2.resize(local_img_left, (368, 368))
            # cv2.imshow("lefthand", Img_left)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
        joint_detections_left = joint_coord_set
    if mean_response_val >= 1:
        tracker.loss_track = False
    else:
        tracker.loss_track = True

def draw_hand(full_img, joint_coords, is_loss_track):
    if not is_loss_track:
    # Plot joints
        for joint_num in range(FLAGS.num_of_joints):
            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                #joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
                cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                           color=(0,0,0), thickness=-1)
            else:
                #joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
                cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                           color=(0, 0, 0), thickness=-1)

    # Plot limbs
        for limb_num in range(len(FLAGS.limbs)):
            x1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][0])
            y1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][1])
            x2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][0])
            y2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][1])
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                #limb_color = list(map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))
                cv2.fillConvexPoly(full_img, polygon, color=(0,255,255))


def draw_hand_flip(full_img, joint_coords, is_loss_track):
    if is_loss_track:
        joint_coords = FLAGS.default_hand
        #print("nothing..")
    # Plot joints
    for joint_num in range(FLAGS.num_of_joints):
        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=1,
                       color=(0,0,0), thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=1,
                       color=(0,0,0), thickness=-1)

    # Plot limbs
    for limb_num in range(len(FLAGS.limbs)):
        x1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][0])
        y1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][1])
        x2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][0])
        y2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][1])
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.fillConvexPoly(full_img, polygon, color=(0,255,255))

def SortPerson(tempX, humans, checkPerson):
    # neck: coordinates of neck-x
    # sorted_neck: coordinates of neck-x sorted in ascending order from left
    neck = np.zeros(len(humans), dtype=int)
    sorted_neck = np.zeros(len(humans), dtype=int)
    for i in range(len(humans)):
        neck[i] = tempX[i][1]
        #print(neck[i])

    for i in range(len(humans)):
        # find the minimum out of all the neck-x
        temp = np.where(neck == min(neck))
        sorted_neck[i] = temp[0][0]

        # change the minimum neck-x to maximum
        neck[sorted_neck[i]] = 9999
    return sorted_neck


# get list exist value
def getExact(array):
    a_len = len(array)               
    if (a_len == 0):
        return 0    
    else:
        if not array:
            array.sort()
        return int(array[a_len-1])

def getMin(array):
    a_len = len(array)
    if (a_len == 0):
        return 0
    else:
        if not array:
            array.sort()
        return int(array[0])

    # get list Median value
def getMedian(array):
    a_len = len(array)               
    if (a_len == 0):
        return 0    
    else:
        for i in range(10):
            if array[i] is None:
                array[i] = 0
        array.sort()
        return int((array[a_len-1] + array[a_len-2])/2)   
        
def Angle_main_calculator(x1,y1,z1,x2,y2,z2):
    v1 = [x1,y1,z1]
    v2 = [x2,y2,z2]
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    #print(angle)
    if not math.isnan(angle) :
        return angle *180./np.pi
        #return int(math.degrees(angle))

# depth data calling
def multiple_depth_data(tempX,tempY,i,humans,depth_data_array,depth_scale):
    for j in range(0,19):
        if tempX[i][j] != 1 and tempY[i][j] != 1: # not int(1), axis X, Y
            temp = []
            minX,minY,maxX,maxY = tempX[i][j]-1,tempY[i][j]-1,tempX[i][j]+2,tempY[i][j]+2
            if minX < 0:
                minX = 0
            if minY < 0:
                minY = 0
            if maxX > 640:
                maxX = 640
            if maxY > 480:
                maxY = 480
            for avgX in range(minX,maxX):
                for avgY in range(minY,maxY):
                    if int(depth_data_array[avgY][avgX]) != 0 and depth_data_array[avgY][avgX]: # array size = 480*640
                        temp.append(int(depth_data_array[avgY][avgX]))
                    # debug print(int((depth_data_array[avgY][avgX]/80)*37.8)) # 1cm = 37.8 pixel
            if temp:
                temp.sort()
                joint_dist[i][j] = int(getMin(temp))  #depth filtering
        # debuging
        # if int(joint_depth_array_temp[i][j]) != 0:
            # cv2.putText(image_rgb,str(int(joint_depth_array_temp[i][j])),(tempX[i][j],tempY[i][j]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    return joint_dist

def hand_depth_data(joint_detections,depth_data_array,hand_joint_dist,check):
    for joint_num in range(FLAGS.num_of_joints):
        if check == 0 :
            depX = int(joint_detections[joint_num][1])
            depY = int(joint_detections[joint_num][0])
        if check == 1:
            depX = int(joint_detections[joint_num][0])
            depY = int(joint_detections[joint_num][1])

        temp = []
        minX,minY,maxX,maxY = depX-5,depY-5,depX+5,depY+5
        if minX < 0:
            minX = 0
        if minY < 0:
            minY = 0
        if maxX > 640:
            maxX = 640
        if maxY > 480:
            maxY = 480
        for avgX in range(minX,maxX):
            for avgY in range(minY,maxY):
                if int(depth_data_array[avgY][avgX]) != 0 and depth_data_array[avgY][avgX]:
                    temp.append(int(depth_data_array[avgY][avgX]))
        if temp:
            temp.sort()
            hand_joint_dist[joint_num] = int(getMin(temp))  #depth filtering
            # First prossecing depth values
            #if tracker.loss_track == False:
            #cv2.putText(local_img, str(hand_joint_dist[joint_num]), (int(local_joint[joint_num][1] - 5), int(local_joint[joint_num][0] - 15)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return hand_joint_dist



def Angle_calulator_hand(hand_Point3dX,hand_Point3dY,hand_Point3dZ,tracker,local_joint,local_img,check):
    angle_filter_hand = [0 for i in range(21)]
    for limb_num in range(len(FLAGS.limbs_angle)):
        if float(hand_Point3dX[int(FLAGS.limbs_angle[limb_num][0])]) != 0 and float(hand_Point3dY[int(FLAGS.limbs_angle[limb_num][0])]) != 0:
            x1=float(hand_Point3dX[int(FLAGS.limbs_angle[limb_num][0])]) - float(hand_Point3dX[int(FLAGS.limbs_angle[limb_num][1])])
            y1=float(hand_Point3dY[int(FLAGS.limbs_angle[limb_num][0])]) - float(hand_Point3dY[int(FLAGS.limbs_angle[limb_num][1])])
            x2=float(hand_Point3dX[int(FLAGS.limbs_angle[limb_num][2])]) - float(hand_Point3dX[int(FLAGS.limbs_angle[limb_num][1])])
            y2=float(hand_Point3dY[int(FLAGS.limbs_angle[limb_num][2])]) - float(hand_Point3dY[int(FLAGS.limbs_angle[limb_num][1])])
            if int(hand_Point3dZ[int(FLAGS.limbs_angle[limb_num][0])]) > 0 :
                z1 = float(hand_Point3dZ[int(FLAGS.limbs_angle[limb_num][0])]) - float(hand_Point3dZ[int(FLAGS.limbs_angle[limb_num][1])])
                z2 = float(hand_Point3dZ[int(FLAGS.limbs_angle[limb_num][2])]) - float(hand_Point3dZ[int(FLAGS.limbs_angle[limb_num][1])])
            else:
                z1, z2 = 0, 0
            flag = int(FLAGS.limbs_angle[limb_num][1])
            if check == 1 :
                angle_filter_hand[flag] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
                cv2.putText(local_img, str(angle_filter_hand[flag]),
                                (int(local_joint[flag][1]), int(local_joint[flag][0])), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
            if check == 0 :
                angle_filter_hand[21-flag] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
                cv2.putText(local_img, str(angle_filter_hand[21-flag]),
                                (int(local_joint[flag][1]), int(local_joint[flag][0])), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 2)
    return angle_filter_hand,local_img




# Not Yet. many peoples, pair of arm angle.
def Angle_calculator(joint_pointX,joint_pointY,joint_dist,angle_filter,angle_filter_count,humans,i):
    if angle_filter_count == 10:
        angle_filter_count = 0

    if humans[i] != None:

        if joint_pointX[i][0] != 1 and joint_pointX[i][1] != 1 and joint_pointX[i][2] != 1:  # Neck 0 , axis 1
            x1 = joint_pointX[i][0] - joint_pointX[i][1]
            y1 = joint_pointY[i][0] - joint_pointY[i][1]
            x2 = joint_pointX[i][2] - joint_pointX[i][1]
            y2 = joint_pointY[i][2] - joint_pointY[i][1]
            if joint_dist[i][0] != 0 and joint_dist[i][1] != 0 and joint_dist[i][2] != 0:
                z1 = joint_dist[i][0] - joint_dist[i][1]
                z2 = joint_dist[i][2] - joint_dist[i][1]
            else:
                z1, z2 = 0, 0
            if joint_pointY[i][1] > joint_pointY[i][2] :
                angle_filter[i][0][angle_filter_count] = -(Angle_main_calculator(x1, y1, z1, x2, y2, z2))
            else :
                angle_filter[i][0][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
            # print(angle_filter[i][0])

            if not angle_filter[i][0][angle_filter_count]:
                angle_filter[i][0][angle_filter_count] = 0
            if angle_filter[i][0][angle_filter_count] > 90:
                angle_filter[i][0][angle_filter_count] = 180 - Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][0][angle_filter_count] = 0

        if joint_pointX[i][1] != 1 and joint_pointX[i][2] != 1 and joint_pointX[i][3] != 1:  # Rshoulder 1 , axis 2
            x1 = joint_pointX[i][1] - joint_pointX[i][2]
            y1 = joint_pointY[i][1] - joint_pointY[i][2]
            x2 = joint_pointX[i][3] - joint_pointX[i][2]
            y2 = joint_pointY[i][3] - joint_pointY[i][2]
            if joint_dist[i][1] != 0 and joint_dist[i][2] != 0 and joint_dist[i][3] != 0:
                z1 = joint_dist[i][1] - joint_dist[i][2]
                z2 = joint_dist[i][3] - joint_dist[i][2]
            else:
                z1, z2 = 0, 0
            angle_filter[i][1][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][1][angle_filter_count] = 0

        if joint_pointX[i][2] != 1 and joint_pointX[i][3] != 1 and joint_pointX[i][4] != 1:  # RElbow 2, axis 3
            x1 = joint_pointX[i][2] - joint_pointX[i][3]
            y1 = joint_pointY[i][2] - joint_pointY[i][3]
            x2 = joint_pointX[i][4] - joint_pointX[i][3]
            y2 = joint_pointY[i][4] - joint_pointY[i][3]
            if joint_dist[i][2] != 0 and joint_dist[i][3] != 0 and joint_dist[i][4] != 0:
                z1 = joint_dist[i][2] - joint_dist[i][3]
                z2 = joint_dist[i][4] - joint_dist[i][3]
            else:
                z1, z2 = 0, 0
            angle_filter[i][2][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][2][angle_filter_count] = 0

        if joint_pointX[i][1] != 1 and joint_pointX[i][5] != 1 and joint_pointX[i][6] != 1:  # Lshoulder 3, axis 5
            x1 = joint_pointX[i][1] - joint_pointX[i][5]
            y1 = joint_pointY[i][1] - joint_pointY[i][5]
            x2 = joint_pointX[i][6] - joint_pointX[i][5]
            y2 = joint_pointY[i][6] - joint_pointY[i][5]
            if joint_dist[i][1] != 0 and joint_dist[i][5] != 0 and joint_dist[i][6] != 0:
                z1 = joint_dist[i][1] - joint_dist[i][5]
                z2 = joint_dist[i][6] - joint_dist[i][5]
            else:
                z1, z2 = 0, 0
            angle_filter[i][3][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][3][angle_filter_count] = 0

        if joint_pointX[i][5] != 1 and joint_pointX[i][6] != 1 and joint_pointX[i][7] != 1:  # LElbow 4, axis 6
            x1 = joint_pointX[i][5] - joint_pointX[i][6]
            y1 = joint_pointY[i][5] - joint_pointY[i][6]
            x2 = joint_pointX[i][7] - joint_pointX[i][6]
            y2 = joint_pointY[i][7] - joint_pointY[i][6]
            if joint_dist[i][5] != 0 and joint_dist[i][6] != 0 and joint_dist[i][7] != 0:
                z1 = joint_dist[i][5] - joint_dist[i][6]
                z2 = joint_dist[i][7] - joint_dist[i][6]
            else:
                z1, z2 = 0, 0
            angle_filter[i][4][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][4][angle_filter_count] = 0

        if joint_pointX[i][1] != 1 and joint_pointX[i][18] != 1 and joint_pointX[i][8] != 1:  # Wrist 5, axis 18
            x1 = joint_pointX[i][1] - joint_pointX[i][18]
            y1 = joint_pointY[i][1] - joint_pointY[i][18]
            x2 = joint_pointX[i][8] - joint_pointX[i][18]
            y2 = joint_pointY[i][8] - joint_pointY[i][18]
            if joint_dist[i][1] != 0 and joint_dist[i][18] != 0 and joint_dist[i][8] != 0:
                z1 = joint_dist[i][1] - joint_dist[i][18]
                z2 = joint_dist[i][8] - joint_dist[i][18]
            else:
                z1, z2 = 0, 0
            angle_filter[i][5][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)

            if not angle_filter[i][5][angle_filter_count]:
                angle_filter[i][5][angle_filter_count] = 0
            if angle_filter[i][5][angle_filter_count] > 90:
                angle_filter[i][5][angle_filter_count] = 180 - Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][5][angle_filter_count] = 0

        if joint_pointX[i][18] != 1 and joint_pointX[i][8] != 1 and joint_pointX[i][9] != 1:  # RHip 6, axis 8
            x1 = joint_pointX[i][18] - joint_pointX[i][8]
            y1 = joint_pointY[i][18] - joint_pointY[i][8]
            x2 = joint_pointX[i][9] - joint_pointX[i][8]
            y2 = joint_pointY[i][9] - joint_pointY[i][8]
            if joint_dist[i][18] != 0 and joint_dist[i][8] != 0 and joint_dist[i][9] != 0:
                z1 = joint_dist[i][18] - joint_dist[i][8]
                z2 = joint_dist[i][9] - joint_dist[i][8]
            else:
                z1, z2 = 0, 0
            angle_filter[i][6][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][6][angle_filter_count] = 0

        if joint_pointX[i][12] != 1 and joint_pointX[i][11] != 1 and joint_pointX[i][18] != 1:  # LHip 7, axis 11
            x1 = joint_pointX[i][12] - joint_pointX[i][11]
            y1 = joint_pointY[i][12] - joint_pointY[i][11]
            x2 = joint_pointX[i][18] - joint_pointX[i][11]
            y2 = joint_pointY[i][18] - joint_pointY[i][11]
            if joint_dist[i][12] != 0 and joint_dist[i][11] != 0 and joint_dist[i][18] != 0:
                z1 = joint_dist[i][12] - joint_dist[i][11]
                z2 = joint_dist[i][18] - joint_dist[i][11]
            else:
                z1, z2 = 0, 0
            angle_filter[i][7][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][7][angle_filter_count] = 0

        if joint_pointX[i][8] != 1 and joint_pointX[i][9] != 1 and joint_pointX[i][10] != 1:  # RKnee 8, axis 9
            x1 = joint_pointX[i][8] - joint_pointX[i][9]
            y1 = joint_pointY[i][8] - joint_pointY[i][9]
            x2 = joint_pointX[i][10] - joint_pointX[i][9]
            y2 = joint_pointY[i][10] - joint_pointY[i][9]
            if joint_dist[i][8] != 0 and joint_dist[i][9] != 0 and joint_dist[i][10] != 0:
                z1 = joint_dist[i][8] - joint_dist[i][9]
                z2 = joint_dist[i][10] - joint_dist[i][9]
            else:
                z1, z2 = 0, 0
            angle_filter[i][8][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][8][angle_filter_count] = 0

        if joint_pointX[i][11] != 1 and joint_pointX[i][12] != 1 and joint_pointX[i][13] != 1:  # LKnee 9, axis 12
            x1 = joint_pointX[i][11] - joint_pointX[i][12]
            y1 = joint_pointY[i][11] - joint_pointY[i][12]
            x2 = joint_pointX[i][13] - joint_pointX[i][12]
            y2 = joint_pointY[i][13] - joint_pointY[i][12]
            if joint_dist[i][11] != 0 and joint_dist[i][12] != 0 and joint_dist[i][13] != 0:
                z1 = joint_dist[i][11] - joint_dist[i][12]
                z2 = joint_dist[i][13] - joint_dist[i][12]
            else:
                z1, z2 = 0, 0
            angle_filter[i][9][angle_filter_count] = Angle_main_calculator(x1, y1, z1, x2, y2, z2)
        else:
            angle_filter[i][9][angle_filter_count] = 0

    angle_filter_count = angle_filter_count + 1
    return angle_filter, angle_filter_count

def Deprojections(tempX, tempY, tempZ,i,humans, depth_intrin, depth_scale):
    for j in range(0, 19):
        temp = [1 for i in range(2)]  # for return value of axisY
        if tempX[i][j] != 1 and tempY[i][j] != 1 and tempZ[i][j] != 1:  # not int(1), axis X, Y
            temp[0] = tempX[i][j]
            temp[1] = tempY[i][j]
            point3d = rs.rs2_deproject_pixel_to_point(depth_intrin, temp, tempZ[i][j] * depth_scale)
            point3dX[i][j] = round(point3d[0], 4)
            point3dY[i][j] = round(point3d[1], 4)
            point3dZ[i][j] = round(point3d[2], 4)
    return point3dX, point3dY, point3dZ

def Deprojections_hand(tempHand, tempZ, depth_intrin, depth_scale,tracker,local_joint,local_img):
    for i in range(0,21):
        temp = [1 for j in range(2)]  # for return value of axisY
        if tempHand[i][0] != 0 and tempHand[i][1] != 0 and tempZ[i] != 0:  # not int(1), axis X, Y
            temp[0] = tempHand[i][0]
            temp[1] = tempHand[i][1]
            point3d = rs.rs2_deproject_pixel_to_point(depth_intrin, temp, tempZ[i] * depth_scale)
            handPoint3dX[i] = round(point3d[0], 4)
            handPoint3dY[i] = round(point3d[1], 4)
            handPoint3dZ[i] = round(point3d[2], 4)
            #if tracker.loss_track == False:
                #cv2.putText(local_img, str(handPoint3dZ[i]), (int(local_joint[i][1] - 5), int(local_joint[i][0] - 15)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # print(handPoint3dZ)
    return handPoint3dX, handPoint3dY, handPoint3dZ

#
# def determineCloserPerson(tempX,tempY,humans,depth_data_array):
#     length = [-1 for j in range(len(humans))]
#     for i in range(len(humans)):
#         if humans[i] != None:
#             if tempX[i][1] == 1: # not int(1), axis X, Y
#                 continue
#             temp = []
#             minX,minY,maxX,maxY = tempX[i][1]-1,tempY[i][1]-1,tempX[i][1]+2,tempY[i][1]+2
#             if minX < 0:
#                 minX = 0
#             if minY < 0:
#                 minY = 0
#             if maxX > 640:
#                 maxX = 640
#             if maxY > 480:
#                 maxY = 480
#             for avgX in range(minX,maxX):
#                 for avgY in range(minY,maxY):
#                     if int(depth_data_array[avgY][avgX]) != 0 and depth_data_array[avgY][avgX]: # array size = 480*640
#                         temp.append(int((depth_data_array[avgY][avgX]/80)*37.8))
#                     # debug print(int((depth_data_array[avgY][avgX]/80)*37.8)) # 1cm = 37.8 pixel
#             if temp:
#                 temp.sort()
#                 length[i]= int(getExact(temp))  #depth filtering
#             # debuging
#             # if int(joint_depth_array_temp[i][j]) != 0:
#             # cv2.putText(image_rgb,str(int(joint_depth_array_temp[i][j])),(tempX[i][j],tempY[i][j]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
#     min = length[0]
#     minPerson = 0
#     for i in range(len(humans)):
#         if min > length[i]:
#             min = length[i]
#             minPerson = i
#
#     return minPerson

def calculate(x1, y1, x2, y2, x3, y3):
    angle = (math.atan2(y1- y2, x1 - x2) - math.atan2(y3 - y2, x3 - x2))
    if not math.isnan(angle) :
        return int(math.degrees(angle))

def calangle(point3dX, point3dY, point3dZ, i) :
    # X: y-z, Y: x-y, Z: x-z
    # RElbow angle
    Y = calculate(point3dX[i][2], point3dY[i][2], point3dX[i][3], point3dY[i][3], point3dX[i][4], point3dY[i][4])
    Z = calculate(point3dX[i][2], point3dZ[i][2], point3dX[i][3], point3dZ[i][3], point3dX[i][4], point3dZ[i][4])
    return Y, Z

# def set_people(joint_pointX, joint_pointY, checkPerson, Set_People):
#     if Set_People:
#         # count found people
#         global last_jointX
#         global last_jointY
#
#         truePeople = checkPerson.nonzero()
#         truePeopleCount = np.shape(truePeople)[1]
#
#         last_jointX = np.zeros((truePeopleCount,19))
#         last_jointY = np.zeros((truePeopleCount,19))
#
#         Set_People = False
#     sorted_neck = SortPerson(joint_pointX, humans, checkPerson)

def sendData2Unity(X, Y, Z, trueHumanCount):
    #### 이제 y, z를 data에 추가하여 전송, unity에서 받은 후 잘라서 활용하기만 하면 됨.
    ##########################################################
    tcpCliSock = socket(AF_INET, SOCK_STREAM)
    try:
        tcpCliSock.connect(ADDR)
    except Exception as e:
    #     print('서버(%s:%s)에 연결 할 수 없습니다.' % ADDR)
        return
    # print('서버(%s:%s)에 연결 되었습니다.' % ADDR)
    ############################################################

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, imgencode = cv2.imencode('.jpg', image_rgb, encode_param)
    data = np.array(imgencode)
    stringData = data.tostring()
    tcpCliSock.send(stringData)
    #print("recv second data", tcpCliSock.recv(100))
    # data 먼저 전송 후 image 전송해야 동작함.
    npz = np.array(Z[0])
    npzero = np.zeros_like(Z[0])
    if not all(npz == npzero):
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4")
        for i in range(0, trueHumanCount):
          x_temp[i] = np.array(X[i])
          # print(x_temp[i,1])
          y_temp[i] = np.array(Y[i])
          z_temp[i] = np.array(Z[i])
        Axis_array = np.hstack(((x_temp*10000), (y_temp*10000), (z_temp*10000)))
        #print(Axis_array)
        Axis_array = Axis_array.astype('int32')
    #print(Axis_array)
        stringData = Axis_array.tostring()
        # print(stringData)
        tcpCliSock.send(stringData)
    #print("recv second data", tcpCliSock.recv(100))


    tcpCliSock.close()

tracker_right = tracking_module.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)
tracker_left = tracking_module.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)
cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)
model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=False)


saver = tf.train.Saver()
output_node = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)
joint_detections_left = np.zeros(shape=(21, 2))
joint_detections_right = np.zeros(shape=(21, 2))
local_joint_right = np.zeros(shape=(21, 2))
local_joint_left = np.zeros(shape=(21, 2))

if __name__ == '__main__':
    # Configure set camera data

    parser = argparse.ArgumentParser(description='SSJointTracker')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default='true',
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    # important calling  (tf-pose-estimation call)
    hand_icon_right = cv2.imread('glove_right.png',cv2.IMREAD_UNCHANGED)
    hand_icon_left = cv2.imread('glove_left.png',cv2.IMREAD_UNCHANGED)
    save_icon = cv2.imread('save.png',cv2.IMREAD_UNCHANGED)
    skeleton_icon = cv2.imread('skeleton.png',cv2.IMREAD_UNCHANGED)
    hand_wating = cv2.imread('hand_inference.png', cv2.IMREAD_UNCHANGED)

    Joint_Record = False
    Set_People = False
    conv = False
    trueHumanCount = 0
    angle_filter_count = 0

    # start camera modules
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()


    #rgb_sensor = profile.get_device()
    # get depth sensor device info

    # setup depth camera long range.
#    if depth_sensor.supports(rs.option.visual_preset):
#        depth_sensor.set_option(rs.option.laser_power, float(1))
#        depth_sensor.set_option(rs.option.visual_preset, float(1))
#        depth_sensor.set_option(rs.option.motion_range, float(65))
#        depth_sensor.set_option(rs.option.confidence_threshold, float(2))

    # can setup move the frame for color of depth
    align_to  = rs.stream.color
    align = rs.align(align_to)
    device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
    sess_config = tf.ConfigProto(device_count=device_count)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess_config.allow_soft_placement = True

    with tf.Session(config=sess_config) as sess:

        model_path_suffix = os.path.join(FLAGS.network_def,
                                         'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                                         'joints_{}'.format(FLAGS.num_of_joints),
                                         'stages_{}'.format(FLAGS.cpm_stages),
                                         'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate,
                                                                          FLAGS.lr_decay_step)
                                         )
        model_save_dir = os.path.join('models',
                                      'weights',
                                      model_path_suffix)
        #print('Load model from [{}]'.format(os.path.join(model_save_dir, FLAGS.model_path)))
        if FLAGS.model_path.endswith('pkl'):
            model.load_weights_from_file(FLAGS.model_path, sess, False)
        else:
            saver.restore(sess, 'C:/tf-SSJointTracker_original/src/models/cpm_hand')
            #saver.restore(sess, 'D:/workspace/project/SSJointTracker/tf-SSJointTracker/src/models/cpm_hand')
        if FLAGS.use_kalman:
            kalman_filter_array_left = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            kalman_filter_array_right = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            for _, joint_kalman_filter_left in enumerate(kalman_filter_array_left):
                joint_kalman_filter_left.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter_left.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter_left.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise
            for _, joint_kalman_filter_right in enumerate(kalman_filter_array_right):
                joint_kalman_filter_right.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter_right.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter_right.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise
        else:
            kalman_filter_array = None

        right_count = 1
        left_count = 1
        right_count_inference = 0
        left_count_inference = 0
        save_count = 0
        pts_count = 0
        while True:
            frames=pipeline.wait_for_frames()
            # one frame reading
            aligned_frames = align.process(frames)
            # can setup move the frame for color of depth
            image_data_rgb = aligned_frames.get_color_frame()
            image_data_depth = aligned_frames.get_depth_frame() # hide image
            # get image module data

            # depth camera extra setting
            #dec_filter = rs.decimation_filter()
            spat_filter = rs.spatial_filter()
            temp_filter = rs.temporal_filter()

            filtered = spat_filter.process(image_data_depth)
            image_data_depth = temp_filter.process(filtered)
            if not image_data_rgb or not image_data_depth:
               continue
            # empty data filtering and flow process

            image_rgb = np.asanyarray(image_data_rgb.get_data())
            depth_data_array = np.asanyarray(image_data_depth.get_data())

            depth_intrin = image_data_depth.profile.as_video_stream_profile().get_intrinsics()
            color_intrin = image_data_rgb.profile.as_video_stream_profile().get_intrinsics()
            depth_to_color_extrin = image_data_depth.profile.get_extrinsics_to(image_data_rgb.profile)

            depth_scale = depth_sensor.get_depth_scale()

            # array format is numpy.
            # save your image data. in numpyarray.
            full_img_right = image_rgb.copy()
            full_img_left = cv2.flip(image_rgb, 1)


            if args.zoom < 1.0:
                canvas = np.zeros_like(image_rgb)
                img_scaled = cv2.resize(image_rgb, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
                dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
                canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
                image_rgb = canvas

            elif args.zoom > 1.0:
                img_scaled = cv2.resize(image_rgb, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (img_scaled.shape[1] - image_rgb.shape[1]) // 2
                dy = (img_scaled.shape[0] - image_rgb.shape[0]) // 2
                image1 = img_scaled[dy:image1.shape[0], dx:image_rgb.shape[1]]
            humans = e.inference(image_rgb)

            left_center = {}
            right_center = {}

            image_rgb_noneHumans = cv2.flip(image_rgb, 1)

            if humans:

            # distance values call (humans list length = people).
            # Nose = 0
            # Neck = 1
            # RShoulder = 2
            # RElbow = 3
            # RWrist = 4
            # LShoulder = 5
            # LElbow = 6
            # LWrist = 7
            # RHip = 8
            # RKnee = 9
            # RAnkle = 10
            # LHip = 11
            # LKnee = 12
            # LAnkle = 13
            # pubis = 18
                res_hand=(320,320)
                if Hand_connect == True :
                    if 4 in humans[0].body_parts.keys():
                        right_wrist = humans[0].body_parts[4]
                        image_h, image_w = image_rgb.shape[:2]
                        right_center = (int(right_wrist.x * image_w + 0.5)), (int(right_wrist.y * image_h + 0.5))
                    if 7 in humans[0].body_parts.keys():
                        left_wrist = humans[0].body_parts[7]
                        image_h, image_w = image_rgb.shape[:2]
                        left_center = (int(640-(left_wrist.x * image_w + 0.5))), (int(left_wrist.y * image_h + 0.5))

                    hand_part = "None"

                    if left_center:
                        hand_part = "left"
                        tracker_left.SetCenter(left_center)
                        test_img_left = tracker_left.tracking_by_joints(full_img_left,joint_detections=joint_detections_left)
                        #FLAGS.default_hand_left = joint_detections_left
                        left_count_inference += 1
                    else:
                        test_img_left = tracker_left.tracking_by_joints(full_img_left,joint_detections=joint_detections_left)

                    if test_img_left is not None:
                        crop_full_scale_left = tracker_left.input_crop_ratio
                        test_img_copy_left = test_img_left.copy()
                        test_img_wb_left = utils.img_white_balance(test_img_left, 5)
                        test_img_input_left = normalize_and_centralize_img(test_img_wb_left)
                        stage_heatmap_np_left = sess.run([output_node],
                                                         feed_dict={model.input_images: test_img_input_left})
                        local_img_left = visualize_result_flip(full_img_left, stage_heatmap_np_left, kalman_filter_array_left,
                                                          tracker_left, crop_full_scale_left,
                                                          test_img_copy_left,left_center)
                        check = 1
                        hand_joint_dist_left = hand_depth_data(joint_detections_left, depth_data_array,hand_joint_dist_left, check)
                        handPoint3dX, handPoint3dY, handPoint3dZ = Deprojections_hand(joint_detections_left,hand_joint_dist_left, depth_intrin,depth_scale, tracker_left,local_joint_left,local_img_left)
                        angle_filter_Lhand,local_img_left = Angle_calulator_hand(handPoint3dX, handPoint3dY, handPoint3dZ, tracker_left,local_joint_left,local_img_left,check)
                        if local_img_left is not None:
                            local_img_left = cv2.resize(local_img_left, res_hand, interpolation=cv2.INTER_AREA)
                            cv2.imshow("test2", local_img_left)
                    if left_center and left_count_inference > 10:
                        Img_left = local_img_left
                        if left_count > 0:
                            left_count = 0
                    else :
                        left_count += 1
                        if left_count == 12:
                            left_count = 0
                            Img_left.fill(0)
                            left_count_inference = 0
                            cv2.putText(Img_left, "Loss track data.", (100, 194),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            # cv2.destroyWindow("lefthand")


                    if right_center:
                        hand_part = "right"
                        tracker_right.SetCenter(right_center)
                        test_img_right = tracker_right.tracking_by_joints(image_rgb,
                                                                          joint_detections=joint_detections_right)
                        right_count_inference += 1
                        # FLAGS.default_hand_right = joint_detections_right
                    else:
                        test_img_right = tracker_right.tracking_by_joints(image_rgb,
                                                                          joint_detections=joint_detections_right)
                    if test_img_right is not None:
                        crop_full_scale_right = tracker_right.input_crop_ratio
                        test_img_copy_right = test_img_right.copy()
                        test_img_wb_right = utils.img_white_balance(test_img_right, 5)
                        test_img_input_right = normalize_and_centralize_img(test_img_wb_right)
                        stage_heatmap_np_right = sess.run([output_node],
                                                          feed_dict={model.input_images: test_img_input_right})
                        local_img_right = visualize_result(image_rgb, stage_heatmap_np_right, kalman_filter_array_right,
                                                           tracker_right, crop_full_scale_right,
                                                           test_img_copy_right,right_center)

                        check = 0
                        hand_joint_dist_right = hand_depth_data(joint_detections_right, depth_data_array,hand_joint_dist_right, check)
                        handPoint3dX, handPoint3dY, handPoint3dZ = Deprojections_hand(joint_detections_right,hand_joint_dist_right, depth_intrin,depth_scale,tracker_right,local_joint_right,local_img_right)

                        angle_filter_Rhand,local_img_right = Angle_calulator_hand(handPoint3dX, handPoint3dY, handPoint3dZ,tracker_right,local_joint_right,local_img_right,check)
                        # print(angle_filter_Rhand[20])
                        # print(angle_filter_Rhand[19])
                        # print(angle_filter_Rhand[18])
                        # print(angle_filter_Rhand[17])
                        if local_img_right is not None:
                            local_img_right = cv2.resize(local_img_right, res_hand, interpolation=cv2.INTER_AREA)
                            cv2.imshow("test1", local_img_right)
                    if right_center and right_count_inference > 10:
                        Img_right = local_img_right
                        if right_count > 0:
                            right_count = 0
                    else:
                        right_count += 1
                        if right_count == 12 :
                            right_count = 0
                            Img_right.fill(0)
                            right_count_inference = 0
                            cv2.putText(Img_right, "Loss track data.", (100, 194),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # cv2.destroyWindow("righthand")
                # print(joint_pixelX)
                image_rgb, checkPerson, trueHumanCount, joint_pixelX, joint_pixelY = TfPoseEstimator.draw_humans(image_rgb, humans, joint_pixelX, joint_pixelY, depth_data_array, imgcopy=False)
                image_rgb = cv2.flip(image_rgb, 1)

                # checkPerson: Determining valid person data
                joint_pointX, joint_pointY = TfPoseEstimator.joint_pointer(image_rgb,humans,imgcopy=False)
                # if len(humans) > 1 :
                #     minPerson = determineCloserPerson(joint_pointX,joint_pointY,humans,depth_data_array)
                # elif len(humans) == 1:
                #     minPerson = 0
                # i = minPerson
                    #for i in range(0,len(humans)):

                if trueHumanCount > 0:
                    for k in range(trueHumanCount):

                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
                        image_bg = image_rgb[20:84, 10+40*k:74+40*k]
                        image_bg = cv2.add(skeleton_icon, image_bg)
                        image_rgb[20:84, 10+40*k:74+40*k] = image_bg
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)

                    # set_people(joint_pointX, joint_pointY, checkPerson, Set_People)
                    # sorted_neck = SortPerson(joint_pointX, humans, checkPerson)
                    for i in range(trueHumanCount):

                        joint_dist = multiple_depth_data(joint_pixelX,joint_pixelY,i,humans,depth_data_array,depth_scale)

                        point3dX, point3dY, point3dZ = Deprojections(joint_pixelX, joint_pixelY, joint_dist, i, humans, depth_intrin, depth_scale)
                        # print("x: " + str(joint_pixelX[0][1]) + " y: " + str(joint_pixelY[0][1]) + " z: " + str(joint_dist[0][1]))

                        if Joint_Record:
                            tbname = "person" + str(dblastnumber) + "_" + str(i + 1)
                            #tbname = "dance" + str(aaa) + "_" + str(bbb)
                            mysql.SQL_INSERT(point3dX[i], point3dY[i], point3dZ[i], tbname)
                        #angle_filter,angle_filter_count = Angle_calculator(joint_pointX,joint_pointY,joint_dist,angle_filter,angle_filter_count,humans,i)
                        #angle_filter, angle_filter_count = Angle_calculator(point3dX, point3dY, point3dZ, angle_filter,angle_filter_count, humans, i)


                        # putText Depth
                        # i: human number, j:joint number
        #               list_depth = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18]
        #                for j in list_depth:
        #                    cv2.putText(image_rgb, str(point3dZ[i][j]), (joint_pixelX[i][j] - 5, joint_pixelY[i][j] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # putText Angle
                        # i: human number, j:angle number, list_angle:putText position(joint number)
                        list_angle = [1, 2, 3, 5, 6, 18, 8, 11, 9, 12]
                        #for j in range(10):
                        #    cv2.putText(image_rgb, str(getMedian(angle_filter[i][j])),
                        #                (-(joint_pixelX[i][list_angle[j]]-640), joint_pixelY[i][list_angle[j]]),
                        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # for unity rotation
                # if not point3dX[i][2] == 1 and not point3dY[i][3] == 1 and not point3dZ[i][4] == 0:
                #     angleY, angleZ = calangle(point3dX, point3dY, point3dZ, i)

                if conv :
                        pts[pts_count] = np.array([point3dX[0][4], point3dY[0][4], point3dZ[0][4]])
                        pts_count += 1

            if not humans:
                image_rgb = image_rgb_noneHumans

            #cv2.putText(image_rgb, "Record: %s" % Joint_Record,(500, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
            # else:
                # print("not find person!")
                # sorted_neck = [0]
                # point3dX = [[1 for i in range(19)] for j in range(4)]
                # point3dY = [[1 for i in range(19)] for j in range(4)]
                # point3dZ = [[0 for i in range(19)] for j in range(4)]
            # print("x: " + str(point3dX[0][1]) + " y: " + str(point3dY[0][1]) + " z: " + str(point3dZ[0][1]))
            # cv2.putText(image_rgb, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            key = cv2.waitKey(1)
            if key == 27:                 # esc key
                pipeline.stop()
                break
            elif key == 83 or key == 115: # S or s key
                if Joint_Record == False:
                    dblastnumber = mysql.SQL_SHOWTABLES(len(humans))
                    Joint_Record = True
                else:
                    Joint_Record = False
            # elif key == 65 or key == 97: # A or a key
            #     aaa += 1
            #     bbb = 1
            # elif key == 66 or key == 98:
            #     bbb += 1
            # elif key == 67 or key == 99:
            #     mysql.SQL_CREATETABLE("dance" + str(aaa) + "_" + str(bbb))
            # elif key == 68 or key == 100: # D or d key
            #     mysql.SQL_DROPTABLE()
            #     mysql.SQL_CREATETABLE()
            #     save_count = 0
            elif key == 72 or key == 104: # H or h key
                if Hand_connect == False:
                    Hand_connect = True
                else :
                    Hand_connect = False
                    hand_info = False
            elif key == 73 or key == 105: # I or i key
                if Unity_control == False:
                    Unity_control = True
                else:
                    Unity_control = False

            elif key == 79 or key == 111: # O or o key
                if option_push == False:
                    option_push = True
                else:
                    option_push = False
                    cv2.destroyWindow('Option')
            elif key == 80 or key == 112: # P or p key
                if not Set_People:
                    Set_People = True

            elif key == 67 or key == 99: # C or c key
                if conv == False:
                    conv = True
                else :
                    conv = False

            elif key == 80 or key == 112: # P or p key
                conv = False
                pts_count = 0
                hull = ConvexHull(pts)
                fig = plt.figure()
                ax = fig.add_subplot(111,projection = "3d")
                ax.plot(pts.T[0],pts.T[1],pts.T[2], "ko")
                for s in hull.simplices:
                    s = np.append(s,s[0])
                    print(s)
                    ax.plot(pts[s,0],pts[s,1],pts[s,2],"r-")
                for i in ["x","y","z"] :
                    eval("ax.set_{:s}label('{:s}')".format(i, i))
                plt.show()

                # pts에 zeros에서 값이 나왔기 때문에 0이 아닌 값만 들어와야 된다!! 0 값 때문에 0부분이 자꾸찍힘....


            if Joint_Record == True :
                save_count += 1
                if save_count > 0 and save_count < 7 :
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGRA)
                    image_bg = image_rgb[20:84, 556:620]
                    image_bg = cv2.add(save_icon, image_bg)
                    image_rgb[20:84, 556:620] = image_bg
                    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)
                elif save_count == 12 :
                    save_count = 0

            if Hand_connect == True :
                if hand_info == True or (left_count == 0 and right_count == 0 and right_count_inference > 5 and left_count_inference > 5):
                    hand_info = True
                    Img_left = cv2.resize(Img_left,(400,400))
                    cv2.putText(Img_left, "LEFT HAND", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    Img_right = cv2.resize(Img_right,(400,400))
                    cv2.putText(Img_right, "RIGHT HAND", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    image_rgb = cv2.resize(image_rgb, (800, 600))
                    Img_marge = np.hstack((Img_left, Img_right))
                    image_full_rgb = np.vstack((image_rgb, Img_marge))
                    cv2.destroyWindow('SSJointTracker_BDMode')
                    cv2.destroyWindow('Inference_hand')
                    cv2.imshow('SSJointTracker_HTMode', image_full_rgb)
                elif hand_info == False:
                    cv2.destroyWindow('SSJointTracker_BDMode')
                    cv2.putText(hand_wating, "Please put your hands on the camera.", (10, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow('Inference_hand', hand_wating)
            else :
                #cv2.putText(image_rgb, "Push the 'o', Help option.", (205, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                image_rgb = cv2.resize(image_rgb, (1024, 768))
                cv2.destroyWindow('Inference_hand')
                cv2.destroyWindow('SSJointTracker_HTMode')
                cv2.imshow('SSJointTracker_BDMode', image_rgb)
            fps_time = time.time()

            if option_push == True :
                option_window = np.zeros(shape=(330,545))
                cv2.putText(option_window, "'S' - Database Save", (130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                cv2.putText(option_window, "'D' - Database Delete", (130, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                cv2.putText(option_window, "'H' - Hand Traking Mode", (130, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                cv2.putText(option_window, "'I' - Send data (for Unity)", (130, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                cv2.putText(option_window, "'O' - Option / Option close", (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                cv2.putText(option_window, "'Esc' - Close SSJoint_Tracker", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                cv2.putText(option_window, "# Recommend using hand mode at close range.", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                cv2.putText(option_window, "# And only one people.", (80, 300),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.imshow('Option', option_window)

            if Unity_control == True:
                sendData2Unity(point3dX, point3dY, point3dZ,trueHumanCount)

            frame_add += 1

            # print("-------")
        cv2.destroyAllWindows()
