import argparse
import logging
import time
import pyrealsense2 as rs
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import math
import importlib
import os
from utils import cpm_utils, tracking_module_th, utils
from config import FLAGS
import tensorflow as tf

def normalize_and_centralize_img(img):
    if FLAGS.color_channel == 'GRAY':
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).reshape((FLAGS.input_size, FLAGS.input_size, 1))

    if FLAGS.normalize_img:
        test_img_input = img / 256.0 - 0.5
        test_img_input = np.expand_dims(test_img_input, axis=0)
    else:
        test_img_input = img - 128.0
        test_img_input = np.expand_dims(test_img_input, axis=0)
    #print("test_img_inputtest_img_inputtest_img_input",test_img_input.shape)
    return test_img_input



def visualize_result_flip(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    demo_stage_heatmaps = []
    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
    last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))
    correct_and_draw_hand_flip(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img)
    return crop_img


def correct_and_draw_hand_flip(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    global joint_detections_flip
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

            # Resize back
            correct_coord /= crop_full_scale

            # Substract padding border
            correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
            correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
            correct_coord[0] += tracker.bbox[0]
            correct_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = correct_coord
    else:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).astype(np.float32)

            local_joint_coord_set[joint_num, :] = joint_coord

            # Resize back
            joint_coord /= crop_full_scale

            # Substract padding border
            joint_coord[0] -= (tracker.pad_boundary[2] / crop_full_scale)
            joint_coord[1] -= (tracker.pad_boundary[0] / crop_full_scale)
            joint_coord[0] += tracker.bbox[0]
            joint_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = joint_coord
    if tracker.loss_track:
        #print("loss_track..")
        joint_coords = FLAGS.default_hand
        draw_hand_flip(crop_img, local_joint_coord_set, tracker.loss_track)
        joint_detections_flip = joint_coord_set
    else:
        draw_hand_flip(full_img, joint_coord_set, tracker.loss_track)
        draw_hand_flip(crop_img, local_joint_coord_set, tracker.loss_track)
        joint_detections_flip = joint_coord_set

    if mean_response_val >= 1:
        tracker.loss_track = False
    else:
        tracker.loss_track = True

    cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
                org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))


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
                       color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=1,
                       color=joint_color, thickness=-1)

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
            cv2.fillConvexPoly(full_img, polygon, color=limb_color)



def visualize_result(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    demo_stage_heatmaps = []
    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
        (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
    last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))
    correct_and_draw_hand(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img)
    return crop_img


def correct_and_draw_hand(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    global joint_detections
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

            # Resize back
            correct_coord /= crop_full_scale

            # Substract padding border
            correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
            correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
            correct_coord[0] += tracker.bbox[0]
            correct_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = correct_coord
    else:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).astype(np.float32)

            local_joint_coord_set[joint_num, :] = joint_coord

            # Resize back
            joint_coord /= crop_full_scale

            # Substract padding border
            joint_coord[0] -= (tracker.pad_boundary[2] / crop_full_scale)
            joint_coord[1] -= (tracker.pad_boundary[0] / crop_full_scale)
            joint_coord[0] += tracker.bbox[0]
            joint_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = joint_coord
    if tracker.loss_track:
        #print("loss_track..")
        joint_coords = FLAGS.default_hand
        draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
        joint_detections = joint_coord_set
    else:
        draw_hand(full_img, joint_coord_set, tracker.loss_track)
        draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
        joint_detections = joint_coord_set

    if mean_response_val >= 1:
        tracker.loss_track = False
    else:
        tracker.loss_track = True

    cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
                org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))


def draw_hand(full_img, joint_coords, is_loss_track):
    if is_loss_track:
        joint_coords = FLAGS.default_hand
        #print("nothing..")
    # Plot joints
    for joint_num in range(FLAGS.num_of_joints):
        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=1,
                       color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=1,
                       color=joint_color, thickness=-1)

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
            cv2.fillConvexPoly(full_img, polygon, color=limb_color)

tracker = tracking_module_th.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)
tracker_flip = tracking_module_th.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)
cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)
model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=False)

saver = tf.train.Saver()
output_node = tf.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)

joint_detections = np.zeros(shape=(21, 2))
joint_detections_flip = np.zeros(shape=(21,2))
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
fps_time = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1)
    #parser.add_argument('--resolution', type=str, default='320x240', help='network input resolution. default=432x368')
    parser.add_argument('--resolution', type=str, default='240x160', help='network input resolution. default=432x368') #pretty good at tracking
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()
    global joint_detections
    global joint_detections_flip



    #logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    rgb_sensor = profile.get_device()
    # setup depth camera long range.
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.laser_power, float(1))
        depth_sensor.set_option(rs.option.visual_preset, float(1))
        #depth_sensor.set_option(rs.option.motion_range, float(65))
        #depth_sensor.set_option(rs.option.confidence_threshold, float(2))

    # can setup move the frame for color of depth
    align_to  = rs.stream.color
    align = rs.align(align_to)
    device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
    sess_config = tf.ConfigProto(device_count=device_count)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    #with 구문을 사용하면 자동으로 구문 블락의 끝에서 close()가 호출된다.
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
            saver.restore(sess, 'models/weights/cpm_hand')

        if FLAGS.use_kalman:
            kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            for _, joint_kalman_filter in enumerate(kalman_filter_array):
                joint_kalman_filter.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise
            kalman_filter_array_flip = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            for _, joint_kalman_filter in enumerate(kalman_filter_array_flip):
                joint_kalman_filter.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise

        else:
            kalman_filter_array = None
            kalman_filter_array_flip = None
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

            # array format is numpy.
            # save your image data. in numpyarray.

            #logger.debug('image preprocess+')
            if args.zoom < 1.0:
                canvas = np.zeros_like(image_rgb)
                img_scaled = cv2.resize(image_rgb, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
                dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
                canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
                image_rgb = canvas
            elif args.zoom > 1.0:
                img_scaled = cv2.resize(image_rgb, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (img_scaled.shape[1] - image.shape[1]) // 2
                dy = (img_scaled.shape[0] - image.shape[0]) // 2
                image_rgb = img_scaled[dy:image.shape[0], dx:image.shape[1]]

            #logger.debug('image process+')
            humans = e.inference(image_rgb)
            rCenter = {}
            lCenter = {}
            if 4 in humans[0].body_parts.keys():
                Rwrist = humans[0].body_parts[4]
                image_h, image_w = image_rgb.shape[:2]
                rCenter = (int(Rwrist.x * image_w + 0.5), int(Rwrist.y * image_h + 0.5))
                print("rCenter: ",rCenter)
            if 7 in humans[0].body_parts.keys():
                lwrist = humans[0].body_parts[7]
                image_h, image_w = image_rgb.shape[:2]
                lCenter = (int(lwrist.x * image_w + 0.5), int(lwrist.y * image_h + 0.5))
                print("lCenter: ",lCenter)
            #추가

            Vtype = "None"
            image_rgb = TfPoseEstimator.draw_humans(image_rgb, humans, imgcopy=False)
            #print("rCenter: ", rCenter)
            if lCenter:
                Vtype = "left"
                tracker.SetCenter(lCenter,Vtype)
                test_img = tracker.tracking_by_joints(image_rgb, Vtype, joint_detections=joint_detections)
            else:
                test_img = tracker.tracking_by_joints(image_rgb, Vtype, joint_detections=joint_detections)


            crop_full_scale = tracker.input_crop_ratio

            test_img_copy = test_img.copy()
            test_img_wb = utils.img_white_balance(test_img, 5)
            test_img_input = normalize_and_centralize_img(test_img_wb)
            #init_op = tf.initialize_all_variables().run(session =sess)
            stage_heatmap_np = sess.run([output_node],
                                                feed_dict={model.input_images: test_img_input})

            local_img = visualize_result(image_rgb, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale,
                                         test_img_copy)
            #추가 끝
            #logger.debug('postprocess+')
            if rCenter:
                Vtype = "right"
                tracker_flip.SetCenter(rCenter, Vtype)
                test_img_flip = tracker_flip.tracking_by_joints(image_rgb, Vtype, joint_detections=joint_detections)

            else:
                test_img_flip = tracker_flip.tracking_by_joints(image_rgb, Vtype, joint_detections=joint_detections)


            crop_full_scale_flip = tracker_flip.input_crop_ratio
            test_img_copy_flip = test_img_flip.copy()

            #test_img_flip = cv2.flip(test_img_copy_flip, 1)

            test_img_wb_flip = utils.img_white_balance(test_img_flip, 5)
            test_img_input_flip = normalize_and_centralize_img(test_img_wb_flip)
            stage_heatmap_np_flip = sess.run([output_node],
                                                feed_dict={model.input_images: test_img_input_flip})
            local_img_flip = visualize_result_flip(image_rgb, stage_heatmap_np_flip, kalman_filter_array_flip, tracker_flip, crop_full_scale_flip,
                                         test_img_copy_flip)






            #logger.debug('show+')
            cv2.putText(image_rgb,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)


            cv2.imshow('tf-pose-estimation result', image_rgb)


            cv2.imshow('left', local_img.astype(np.uint8))
            cv2.imshow('right', local_img_flip.astype(np.uint8))
            fps_time = time.time()
            #esc 키 누를시 break;
            if cv2.waitKey(1) == 27:
                break
        #logger.debug('finished+')

    cv2.destroyAllWindows()
