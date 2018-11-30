import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
out = cv2.VideoWriter('mycam.avi', fcc, 7, (640, 480))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='D:\workspace\project\study\python\\mycam.avi')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default='true',
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    # logger.debug('cam read+')
    # cam = cv2.VideoCapture(args.camera)
    cap = cv2.VideoCapture(args.video)
    # ret_val, image = cap.read()
    # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    while (cap.isOpened()):
        ret_val, image = cap.read()

        humans = e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # logger.debug('show+')
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        res = (640, 480)
        image = cv2.resize(image, res, interpolation=cv2.INTER_AREA)
        cv2.imshow('tf-pose-estimation result', image)
        out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()
logger.debug('finished+')
