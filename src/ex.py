import logging
logging.basicConfig(level=logging.INFO)
import numpy as np 
 
import matplotlib.pyplot as plt
import pyrealsense as pyrs
from pyrealsense import rsutilwrapper
from ipdb import set_trace
 
depth_point = np.zeros(3, dtype=np.float32)
color_point = np.zeros(3, dtype=np.float32)
 
def rs_transform_point_to_point(to_point, extrin, from_point):
    to_point[0] = extrin.rotation[0] * from_point[0] + extrin.rotation[3] * from_point[1] + extrin.rotation[6] * from_point[2] + extrin.translation[0]
    to_point[1] = extrin.rotation[1] * from_point[0] + extrin.rotation[4] * from_point[1] + extrin.rotation[7] * from_point[2] + extrin.translation[1]
    to_point[2] = extrin.rotation[2] * from_point[0] + extrin.rotation[5] * from_point[1] + extrin.rotation[8] * from_point[2] + extrin.translation[2]
 
with pyrs.Service():
    dev = pyrs.Device()
    extrinsics = dev.get_device_extrinsics(dev.streams[1].stream, dev.streams[0].stream)
    dev.wait_for_frames()
     
    while True:
        c = dev.color
        temp = dev.depth
         
        cad = dev.cad
 
 
        plt.imshow(temp)
        plt.show()
 
        print("creating aligned depth image of shape", temp.shape)
        depth_point = np.zeros(3, dtype=np.float32)
         
        depth_aligned = np.zeros(temp.shape)
        for y in range(temp.shape[0]):
            for x in range(temp.shape[1]):
                depth_in_meters = temp[y,x]*dev.depth_scale
                depth_pixel = np.array([x,y])
                rsutilwrapper.deproject_pixel_to_point(depth_point, dev.depth_intrinsics, depth_pixel, depth_in_meters)
                rs_transform_point_to_point(color_point, extrinsics, depth_point)
                pixel = np.ones(2, dtype=np.float32) * np.NaN
                rsutilwrapper.project_point_to_pixel( pixel, dev.color_intrinsics, color_point)
                pixel[0] += temp.shape[1]/2.0
                pixel[1] += temp.shape[0]/2.0
                pixel = pixel.astype(np.uint16)
                if ( (pixel[0]>= 0) and (pixel[0] < temp.shape[1]) and (pixel[1]>= 0) and (pixel[1] < temp.shape[0])):
                    depth_aligned[pixel[1],pixel[0]] = depth_in_meters
                    print(pixel, depth_point)
                # if temp[y,x]>0:
        set_trace()
