import pyrealsense2 as rs
# pyrealsense sdk 2.0ver import
import math

import argparse
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

fps_time = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# pipeline is that import realsense 3 modules.
# config streaming values set your camera configure.
# enable_stream(camera_module, resolution_w,resolution_h,pixel_format,frame)
# frame was fixed values.  

joint_dist = [[0 for i in range(19)] for j in range(12)] # distance array values setup
joint_pointX = [[1 for i in range(19)] for j in range(12)] # for return value of axisX
joint_pointY = [[1 for i in range(19)] for j in range(12)] # for return value of axisY
joint_depth_array_temp = [[0 for i in range(19)] for j in range(12)]
frame_add = 0
angle_filter = [[[0 for i in range(10)] for j in range(10)] for l in range(12)]
angle_filter_count=0
coordinates=[[0 for i in range(19)]]

point3dX = [[1 for i in range(19)] for j in range(12)] # for return value of axisX
point3dY = [[1 for i in range(19)] for j in range(12)] # for return value of axisY
point3dZ = [[0 for i in range(19)] for j in range(12)] # distance array values setup

# get list exist value
def getExact(array):
    a_len = len(array)               
    if (a_len == 0):
        return 0    
    else:
        if not array:
            array.sort()
        return int(array[a_len-1])   


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
    cal1 = ((x1*x2) + (y1*y2) + (z1*z2))
    cal2 = math.sqrt(math.pow(x1, 2) + math.pow(y1, 2) + math.pow(z1, 2)) * math.sqrt(pow(x2, 2) + math.pow(y2, 2) + math.pow(z2, 2))
    #print("cal1: "+str(cal1)+" cal2: " + str(cal2))
    #print("x1:" + str(x1) + " y1:"+str(y1)+" z1"+str(z1)+" x2"+str(x2)+" y2"+str(y2)+" z2"+str(z2))
    if cal2 != 0:
        return int(math.acos(cal1 / cal2) * 180 / math.pi)

# depth data calling
def multiple_depth_data(tempX,tempY,i,humans,depth_data_array,image_rgb,depth_scale):
    for j in range(0,14):
        if humans[i] != None: 
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
                    joint_depth_array_temp[i][j] = int(getExact(temp))  #depth filtering
            # debuging
            # if int(joint_depth_array_temp[i][j]) != 0:
                # cv2.putText(image_rgb,str(int(joint_depth_array_temp[i][j])),(tempX[i][j],tempY[i][j]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

    

    return joint_depth_array_temp,image_rgb

#def 2dTo3d()


 # Not Yet. many peoples, pair of arm angle. 
def Angle_calculator(joint_pointX,joint_pointY,joint_dist, angle_filter,angle_filter_count,humans,i):

    if angle_filter_count == 10:
        angle_filter_count = 0

    if humans[i] != None:

        if joint_pointX[i][0] != 1 and joint_pointX[i][1] != 1 and joint_pointX[i][2] != 1  : # Neck 0 , axis 1
            x1=joint_pointX[i][0] - joint_pointX[i][1]
            y1=joint_pointY[i][0] - joint_pointY[i][1]
            x2=joint_pointX[i][2] - joint_pointX[i][1]
            y2=joint_pointY[i][2] - joint_pointY[i][1]
            if joint_dist[i][0] != 0 and joint_dist[i][1] != 0 and joint_dist[i][2] != 0 :
                z1=joint_dist[i][0] - joint_dist[i][1]
                z2=joint_dist[i][2] - joint_dist[i][1]
            else :
                z1,z2=0,0 
            angle_filter[i][0][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
            # print(angle_filter[i][0])

            if not angle_filter[i][0][angle_filter_count] :
               angle_filter[i][0][angle_filter_count] = 0
            if angle_filter[i][0][angle_filter_count] > 90 :
               angle_filter[i][0][angle_filter_count] = 180 - Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][0][angle_filter_count] = 0


        if joint_pointX[i][1] != 1 and joint_pointX[i][2] != 1 and joint_pointX[i][3] != 1  : # Rshoulder 1 , axis 2
            x1=joint_pointX[i][1] - joint_pointX[i][2]
            y1=joint_pointY[i][1] - joint_pointY[i][2]
            x2=joint_pointX[i][3] - joint_pointX[i][2]
            y2=joint_pointY[i][3] - joint_pointY[i][2]
            if joint_dist[i][1] != 0 and joint_dist[i][2] != 0 and joint_dist[i][3] != 0 :
                z1=joint_dist[i][1] - joint_dist[i][2]
                z2=joint_dist[i][3] - joint_dist[i][2]
            else :
                z1,z2=0,0
            angle_filter[i][1][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][1][angle_filter_count] = 0


        if joint_pointX[i][2] != 1 and joint_pointX[i][3] != 1 and joint_pointX[i][4] != 1  : # RElbow 2, axis 3
            x1=joint_pointX[i][2] - joint_pointX[i][3]
            y1=joint_pointY[i][2] - joint_pointY[i][3]
            x2=joint_pointX[i][4] - joint_pointX[i][3]
            y2=joint_pointY[i][4] - joint_pointY[i][3]
            if joint_dist[i][2] != 0 and joint_dist[i][3] != 0 and joint_dist[i][4] != 0 :
                z1=joint_dist[i][2] - joint_dist[i][3]
                z2=joint_dist[i][4] - joint_dist[i][3]
            else :
                z1,z2=0,0
            angle_filter[i][2][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][2][angle_filter_count] = 0

        
        if joint_pointX[i][1] != 1 and joint_pointX[i][5] != 1 and joint_pointX[i][6] != 1 : # Lshoulder 3, axis 5
            x1=joint_pointX[i][1] - joint_pointX[i][5]
            y1=joint_pointY[i][1] - joint_pointY[i][5]
            x2=joint_pointX[i][6] - joint_pointX[i][5]
            y2=joint_pointY[i][6] - joint_pointY[i][5]
            if joint_dist[i][1] != 0 and joint_dist[i][5] != 0 and joint_dist[i][6] != 0 :
                z1=joint_dist[i][1] - joint_dist[i][5]
                z2=joint_dist[i][6] - joint_dist[i][5]
            else :
                z1,z2=0,0
            angle_filter[i][3][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][3][angle_filter_count] = 0

        if joint_pointX[i][5] != 1 and joint_pointX[i][6] != 1 and joint_pointX[i][7] != 1 : # LElbow 4, axis 6
            x1=joint_pointX[i][5] - joint_pointX[i][6]
            y1=joint_pointY[i][5] - joint_pointY[i][6]
            x2=joint_pointX[i][7] - joint_pointX[i][6]
            y2=joint_pointY[i][7] - joint_pointY[i][6]
            if joint_dist[i][5] != 0 and joint_dist[i][6] != 0 and joint_dist[i][7] != 0 :
                z1=joint_dist[i][5] - joint_dist[i][6]
                z2=joint_dist[i][7] - joint_dist[i][6]
            else :
                z1,z2=0,0
            angle_filter[i][4][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][4][angle_filter_count] = 0

        if joint_pointX[i][1] != 1 and joint_pointX[i][18] != 1 and joint_pointX[i][8] != 1  : # Wrist 5, axis 18
            x1=joint_pointX[i][1] - joint_pointX[i][18]
            y1=joint_pointY[i][1] - joint_pointY[i][18]
            x2=joint_pointX[i][8] - joint_pointX[i][18]
            y2=joint_pointY[i][8] - joint_pointY[i][18]
            if joint_dist[i][1] != 0 and joint_dist[i][18] != 0 and joint_dist[i][8] != 0 :
                z1=joint_dist[i][1] - joint_dist[i][18]
                z2=joint_dist[i][8] - joint_dist[i][18]
            else :
                z1,z2=0,0 
            angle_filter[i][5][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)

            if not angle_filter[i][5][angle_filter_count] :
               angle_filter[i][5][angle_filter_count] = 0
            if angle_filter[i][5][angle_filter_count] > 90 :
               angle_filter[i][5][angle_filter_count] = 180 - Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][5][angle_filter_count] = 0

        if joint_pointX[i][18] != 1 and joint_pointX[i][8] != 1 and joint_pointX[i][9] != 1 : # RHip 6, axis 8
            x1=joint_pointX[i][18] - joint_pointX[i][8]
            y1=joint_pointY[i][18] - joint_pointY[i][8]
            x2=joint_pointX[i][9] - joint_pointX[i][8]
            y2=joint_pointY[i][9] - joint_pointY[i][8]
            if joint_dist[i][18] != 0 and joint_dist[i][8] != 0 and joint_dist[i][9] != 0 :
                z1=joint_dist[i][18] - joint_dist[i][8]
                z2=joint_dist[i][9] - joint_dist[i][8]
            else :
                z1,z2=0,0
            angle_filter[i][6][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][6][angle_filter_count] = 0

        if joint_pointX[i][12] != 1 and joint_pointX[i][11] != 1 and joint_pointX[i][18] != 1 : # LHip 7, axis 11
            x1=joint_pointX[i][12] - joint_pointX[i][11]
            y1=joint_pointY[i][12] - joint_pointY[i][11]
            x2=joint_pointX[i][18] - joint_pointX[i][11]
            y2=joint_pointY[i][18] - joint_pointY[i][11]
            if joint_dist[i][12] != 0 and joint_dist[i][11] != 0 and joint_dist[i][18] != 0 :
                z1=joint_dist[i][12] - joint_dist[i][11]
                z2=joint_dist[i][18] - joint_dist[i][11]
            else :
                z1,z2=0,0
            angle_filter[i][7][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][7][angle_filter_count] = 0

        if joint_pointX[i][8] != 1 and joint_pointX[i][9] != 1 and joint_pointX[i][10] != 1 : # RKnee 8, axis 9
            x1=joint_pointX[i][8] - joint_pointX[i][9]
            y1=joint_pointY[i][8] - joint_pointY[i][9]
            x2=joint_pointX[i][10] - joint_pointX[i][9]
            y2=joint_pointY[i][10] - joint_pointY[i][9]
            if joint_dist[i][8] != 0 and joint_dist[i][9] != 0 and joint_dist[i][10] != 0 :
                z1=joint_dist[i][8] - joint_dist[i][9]
                z2=joint_dist[i][10] - joint_dist[i][9]
            else :
                z1,z2=0,0
            angle_filter[i][8][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][8][angle_filter_count] = 0

        if joint_pointX[i][11] != 1 and joint_pointX[i][12] != 1 and joint_pointX[i][13] != 1 : # LKnee 9, axis 12
            x1=joint_pointX[i][11] - joint_pointX[i][12]
            y1=joint_pointY[i][11] - joint_pointY[i][12]
            x2=joint_pointX[i][13] - joint_pointX[i][12]
            y2=joint_pointY[i][13] - joint_pointY[i][12]
            if joint_dist[i][11] != 0 and joint_dist[i][12] != 0 and joint_dist[i][13] != 0 :
                z1=joint_dist[i][11] - joint_dist[i][12]
                z2=joint_dist[i][13] - joint_dist[i][12]
            else :
                z1,z2=0,0
            angle_filter[i][9][angle_filter_count] = Angle_main_calculator(x1,y1,z1,x2,y2,z2)
        else:
            angle_filter[i][9][angle_filter_count] = 0


       
    angle_filter_count = angle_filter_count + 1
    return angle_filter,angle_filter_count

def Deprojections(tempX, tempY, tempZ,i,humans, depth_intrin, depth_scale):
    for j in range(0, 14):
        if humans[i] != None:
            temp = [1 for i in range(2)]  # for return value of axisY
            if tempX[i][j] != 1 and tempY[i][j] != 1 and tempZ[i][j] != 1:  # not int(1), axis X, Y
                temp[0] = tempX[i][j]
                temp[1] = tempY[i][j]
                point3d = rs.rs2_deproject_pixel_to_point(depth_intrin, temp, tempZ[i][j] * depth_scale)
                point3dX[i][j] = round(point3d[0], 4)
                point3dY[i][j] = round(point3d[1], 4)
                point3dZ[i][j] = round(point3d[2], 4)
    return point3dX, point3dY, point3dZ

if __name__ == '__main__':
    # Configure set camera data
    parser = argparse.ArgumentParser(description='SSJointTracker')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default='true',
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    # important calling  (tf-pose-estimation call)

    profile = pipeline.start(config)
    # start camera modules

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

    #f = open("D:\ho\바탕화면\\newfile.txt", 'w')
    while True:

        frames=pipeline.wait_for_frames()
        # one frame reading
        aligned_frames = align.process(frames)
        # can setup move the frame for color of depth
        image_data_rgb = aligned_frames.get_color_frame()
        image_data_depth = aligned_frames.get_depth_frame() # hide image
        # get image module data

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
        image_rgb = TfPoseEstimator.draw_humans(image_rgb,humans,imgcopy=False)

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
        # Wrist = 18


            joint_pointX, joint_pointY = TfPoseEstimator.joint_pointer(image_rgb,humans,imgcopy=False)
            if len(humans) > 0 :
                for i in range(0,len(humans)):

                    joint_dist,image_rgb = multiple_depth_data(joint_pointX,joint_pointY,i,humans,depth_data_array,image_rgb,depth_scale)

                    point3dX, point3dY, point3dZ = Deprojections(joint_pointX, joint_pointY, joint_dist, i, humans, depth_intrin, depth_scale)

                    length =round(math.sqrt(math.pow(point3dX[i][0] - point3dX[i][1],2) + math.pow(point3dY[i][0] - point3dY[i][1],2) + math.pow(point3dZ[i][0] - point3dZ[i][1],2)), 4)
                    print(length)

                    #angle_filter,angle_filter_count = Angle_calculator(joint_pointX,joint_pointY,joint_dist,angle_filter,angle_filter_count,humans,i)
                    angle_filter, angle_filter_count = Angle_calculator(point3dX, point3dY, point3dZ,angle_filter, angle_filter_count, humans, i)

                    # write file - frame, x, y, z
                    #for j in range(0,19):
                    #data = "x:"+str(joint_pointX[i][j])+" y:"+str(joint_pointY[i][j])+" z:"+str(joint_dist[i][j])+"\n"
                    data = "fps:" + str(frame_add) +"-"+ str(i)+" x:" + str(joint_pointX[i][0]) + " y:" + str(joint_pointY[i][0]) + " z:" + str(joint_dist[i][0]) + "\n"
                    #f.write(data)

                    text = str(length) #+ "\nz1:"+ str(point3dZ[i][2]) + "\nz2"+str(point3dZ[i][5])
                    #print("x: ",point3dX[i][0], " y: ",point3dY[i][0], " z :", point3dZ[i][0])
                    #cv2.putText(image_rgb, text, (20,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
                    #cv2.putText(image_rgb, text, (joint_pointX[i][0], joint_pointY[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # putText Depth
                    cv2.putText(image_rgb, str(point3dZ[i][0]), (joint_pointX[i][0] - 5, joint_pointY[i][0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][1]), (joint_pointX[i][1] - 5, joint_pointY[i][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][2]), (joint_pointX[i][2] - 5, joint_pointY[i][2] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][3]), (joint_pointX[i][3] - 5, joint_pointY[i][3] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][4]), (joint_pointX[i][4] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][5]), (joint_pointX[i][5] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][6]), (joint_pointX[i][6] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][7]), (joint_pointX[i][7] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][8]), (joint_pointX[i][8] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][9]), (joint_pointX[i][9] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][10]), (joint_pointX[i][10] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][11]), (joint_pointX[i][4] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][12]), (joint_pointX[i][4] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][13]), (joint_pointX[i][4] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(image_rgb, str(point3dZ[i][14]), (joint_pointX[i][4] - 5, joint_pointY[i][4] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # putText Angle
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][0])),(joint_pointX[i][1],joint_pointY[i][1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][1])),(joint_pointX[i][2],joint_pointY[i][2]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][2])),(joint_pointX[i][3],joint_pointY[i][3]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][3])),(joint_pointX[i][5],joint_pointY[i][5]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][4])),(joint_pointX[i][6],joint_pointY[i][6]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][5])),(joint_pointX[i][18],joint_pointY[i][18]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][6])),(joint_pointX[i][8],joint_pointY[i][8]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][7])),(joint_pointX[i][9],joint_pointY[i][9]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][8])),(joint_pointX[i][11],joint_pointY[i][11]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
                    cv2.putText(image_rgb,str(getMedian(angle_filter[i][9])),(joint_pointX[i][12],joint_pointY[i][12]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

        cv2.putText(image_rgb,
            "FPS: %f" % (1.0 / (time.time() - fps_time)),
            (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)
        res=(1280,960)
        image_rgb = cv2.resize(image_rgb,res,interpolation=cv2.INTER_AREA)
        cv2.imshow('SSJointTracker', image_rgb)
        
        fps_time = time.time()    

        if cv2.waitKey(1) == 27:
            pipeline.stop()
            break

        frame_add += 1    

        print("-------")
    #f.close()
    cv2.destroyAllWindows()
