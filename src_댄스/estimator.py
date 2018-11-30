import itertools
import logging
import math
from collections import namedtuple

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import maximum_filter, gaussian_filter

import common
from common import CocoPairsNetwork, CocoPairs, CocoPart

logger = logging.getLogger('SSJointTracker')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """

    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)


class PoseEstimator:
    heatmap_supress = False
    heatmap_gaussian = False
    adaptive_threshold = False

    NMS_Threshold = 0.15
    Local_PAF_Threshold = 0.2
    PAF_Count_Threshold = 5
    Part_Count_Threshold = 4
    Part_Score_Threshold = 4.5

    PartPair = namedtuple('PartPair', [
        'score',
        'part_idx1', 'part_idx2',
        'idx1', 'idx2',
        'coord1', 'coord2',
        'score1', 'score2'
    ], verbose=False)

    def __init__(self):
        pass

    @staticmethod
    def non_max_suppression(plain, window_size=3, threshold=NMS_Threshold):
        under_threshold_indices = plain < threshold
        plain[under_threshold_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((window_size, window_size))))

    @staticmethod
    def estimate(heat_mat, paf_mat):
        if heat_mat.shape[2] == 19:
            heat_mat = np.rollaxis(heat_mat, 2, 0)
        if paf_mat.shape[2] == 38:
            paf_mat = np.rollaxis(paf_mat, 2, 0)
        #print(heat_mat.shape[2])
        if PoseEstimator.heatmap_supress:
            heat_mat = heat_mat - heat_mat.min(axis=1).min(axis=1).reshape(19, 1, 1)
            heat_mat = heat_mat - heat_mat.min(axis=2).reshape(19, heat_mat.shape[1], 1)

        if PoseEstimator.heatmap_gaussian:
            heat_mat = gaussian_filter(heat_mat, sigma=0.5)

        if PoseEstimator.adaptive_threshold:
            _NMS_Threshold = max(np.average(heat_mat) * 4.0, PoseEstimator.NMS_Threshold)
            _NMS_Threshold = min(_NMS_Threshold, 0.3)
        else:
            _NMS_Threshold = PoseEstimator.NMS_Threshold

        # extract interesting coordinates using NMS.
        coords = []     # [[coords in plane1], [....], ...]
        for plain in heat_mat[:-1]:
            nms = PoseEstimator.non_max_suppression(plain, 5, _NMS_Threshold)
            coords.append(np.where(nms >= _NMS_Threshold))

        # score pairs
        pairs_by_conn = list()
        for (part_idx1, part_idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
            pairs = PoseEstimator.score_pairs(
                part_idx1, part_idx2,
                coords[part_idx1], coords[part_idx2],
                paf_mat[paf_x_idx], paf_mat[paf_y_idx],
                heatmap=heat_mat,
                rescale=(1.0 / heat_mat.shape[2], 1.0 / heat_mat.shape[1])
            )

            pairs_by_conn.extend(pairs)

        # merge pairs to human
        # pairs_by_conn is sorted by CocoPairs(part importance) and Score between Parts.
        humans = [Human([pair]) for pair in pairs_by_conn]
        while True:
            merge_items = None
            for k1, k2 in itertools.combinations(humans, 2):
                if k1 == k2:
                    continue
                if k1.is_connected(k2):
                    merge_items = (k1, k2)
                    break

            if merge_items is not None:
                merge_items[0].merge(merge_items[1])
                humans.remove(merge_items[1])
            else:
                break

        # reject by subset count
        humans = [human for human in humans if human.part_count() >= PoseEstimator.PAF_Count_Threshold]

        # reject by subset max score
        humans = [human for human in humans if human.get_max_score() >= PoseEstimator.Part_Score_Threshold]

        return humans

    @staticmethod
    def score_pairs(part_idx1, part_idx2, coord_list1, coord_list2, paf_mat_x, paf_mat_y, heatmap, rescale=(1.0, 1.0)):
        connection_temp = []

        cnt = 0
        for idx1, (y1, x1) in enumerate(zip(coord_list1[0], coord_list1[1])):
            for idx2, (y2, x2) in enumerate(zip(coord_list2[0], coord_list2[1])):
                score, count = PoseEstimator.get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y)
                cnt += 1
                if count < PoseEstimator.PAF_Count_Threshold or score <= 0.0:
                    continue
                connection_temp.append(PoseEstimator.PartPair(
                    score=score,
                    part_idx1=part_idx1, part_idx2=part_idx2,
                    idx1=idx1, idx2=idx2,
                    coord1=(x1 * rescale[0], y1 * rescale[1]),
                    coord2=(x2 * rescale[0], y2 * rescale[1]),
                    score1=heatmap[part_idx1][y1][x1],
                    score2=heatmap[part_idx2][y2][x2],
                ))

        connection = []
        used_idx1, used_idx2 = set(), set()
        for candidate in sorted(connection_temp, key=lambda x: x.score, reverse=True):
            # check not connected
            if candidate.idx1 in used_idx1 or candidate.idx2 in used_idx2:
                continue
            connection.append(candidate)
            used_idx1.add(candidate.idx1)
            used_idx2.add(candidate.idx2)

        return connection

    @staticmethod
    def get_score(x1, y1, x2, y2, paf_mat_x, paf_mat_y):
        __num_inter = 10
        __num_inter_f = float(__num_inter)
        dx, dy = x2 - x1, y2 - y1
        normVec = math.sqrt(dx ** 2 + dy ** 2)

        if normVec < 1e-4:
            return 0.0, 0

        vx, vy = dx / normVec, dy / normVec

        xs = np.arange(x1, x2, dx / __num_inter_f) if x1 != x2 else np.full((__num_inter,), x1)
        ys = np.arange(y1, y2, dy / __num_inter_f) if y1 != y2 else np.full((__num_inter,), y1)
        xs = (xs + 0.5).astype(np.int8)
        ys = (ys + 0.5).astype(np.int8)

        # without vectorization
        pafXs = np.zeros(__num_inter)
        pafYs = np.zeros(__num_inter)
        for idx, (mx, my) in enumerate(zip(xs, ys)):
            pafXs[idx] = paf_mat_x[my][mx]
            pafYs[idx] = paf_mat_y[my][mx]

        # vectorization slow?
        # pafXs = pafMatX[ys, xs]
        # pafYs = pafMatY[ys, xs]

        local_scores = pafXs * vx + pafYs * vy
        thidxs = local_scores > PoseEstimator.Local_PAF_Threshold

        return sum(local_scores * thidxs), sum(thidxs)


class TfPoseEstimator:
    ENSEMBLE = 'addup'        # average, addup

    def __init__(self, graph_path, target_size=(320, 240)):
        self.target_size = target_size

        # load graph
        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name='TfPoseEstimator')
        tf.device('/gpu:0')
        self.persistent_sess = tf.Session(graph=self.graph)

        # for op in self.graph.get_operations():
        #     print(op.name)

        self.tensor_image = self.graph.get_tensor_by_name('TfPoseEstimator/image:0')
        self.tensor_output = self.graph.get_tensor_by_name('TfPoseEstimator/Openpose/concat_stage7:0')

        self.heatMat = self.pafMat = None

        # warm-up
        self.persistent_sess.run(
            self.tensor_output,
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)]
            }
        )

    def __del__(self):
        self.persistent_sess.close()

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2**8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q


    @staticmethod
    def draw_humans(npimg, humans, joint_pixelX, joint_pixelY, depth_data_array, imgcopy=False):
        rgb = [[0, 228, 255], [22, 219, 29], [255, 84, 0],[255, 0, 95]]

        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        centers = {}

        weight = 0.6
        humanCount = len(humans)
        checkPerson = np.ones(humanCount, dtype=bool)
        neck = np.zeros(humanCount, dtype=int)

        k = -1
        for human in humans:
            k += 1
            if len(human.body_parts) < 10:
                checkPerson[k] = 0
                neck[k] = 9999
                # print("continue")
                continue
            # neck - x 추출, 대입, 정렬

            neck[k] = int(depth_data_array[int(human.body_parts[1].y * image_h + 0.5)][int(human.body_parts[1].x * image_w +0.5)])

        trueHuman = checkPerson.nonzero()
        trueHumanCount =np.shape(trueHuman)[1]

        # index of the neck-x in ascending sort order
        sorted_neck_index = np.zeros(trueHumanCount, dtype=int)
        sorted_neck = np.zeros(trueHumanCount, dtype=int)
        for i in range(trueHumanCount):
            # find the minimum out of all the neck-x
            temp = np.where(neck == min(neck))
            sorted_neck_index[i] = temp[0][0]
            sorted_neck[i] = neck[sorted_neck_index[i]]
            # change the minimum neck-x to maximum
            neck[sorted_neck_index[i]] = 9999

        # 정렬된 순서로 draw circle/ line
        for j in range(trueHumanCount):
            # if j >0:
            #     continue

            # draw point
            wrist_inference = (0, 0)
            wrist_check = 0
            index = sorted_neck_index[j]
            human = humans[index]

            section = 160
            # human is in section
            # if sorted_neck[j] >= section*(j) and sorted_neck[j] <section * (j+1):
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys() or i > 13 :
                    continue

                body_part = human.body_parts[i]

                if not joint_pixelX[j][i] == 1:
                    joint_pixelX[j][i] = int(joint_pixelX[j][i] * weight + int(body_part.x * image_w + 0.5) * (1 - weight))
                    joint_pixelY[j][i] = int(joint_pixelY[j][i] * weight + int(body_part.y * image_h + 0.5) * (1 - weight))
                else:
                    joint_pixelX[j][i] = int(body_part.x * image_w + 0.5)
                    joint_pixelY[j][i] = int(body_part.y * image_h + 0.5)
            for i in range(common.CocoPart.Background.value):
                if i not in human.body_parts.keys() or i > 13:
                    continue

                # body_part = human.body_parts[i]
                human.body_parts[i].x = joint_pixelX[j][i]
                human.body_parts[i].y = joint_pixelY[j][i]
                body_part = human.body_parts[i]
                center = (body_part.x, body_part.y)
                centers[i] = center
                cv2.circle(npimg, center, 1, (0, 0, 0), thickness=6, lineType=1, shift=0)


                if i == 8:
                    wrist_inference = center

                if wrist_inference != (0,0) and i == 11:
                    wrist_inference = (int((wrist_inference[0] + center[0]) / 2),int((wrist_inference[1] + center[1]) / 2))
                    cv2.circle(npimg, wrist_inference, 1, (0,0,0), thickness=6, lineType=1, shift=0)
                    wrist_check = 1
                    centers[18] = wrist_inference

                '''  Part Test
                if i==0:
                    cv2.putText(npimg,"nose",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==1:
                    cv2.putText(npimg,"neck",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==2:
                    cv2.putText(npimg,"Rshoulder",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==3:
                    cv2.putText(npimg,"RElbow",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==4:
                    cv2.putText(npimg,"RWrist",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==5:
                    cv2.putText(npimg,"LShoulder",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==6:
                    cv2.putText(npimg,"LElbow",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==7:
                    cv2.putText(npimg,"LWrist",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==8:
                    cv2.putText(npimg,"RHip",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==9:
                    cv2.putText(npimg,"RKnee",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==10:
                    cv2.putText(npimg,"RAnkle",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==11:
                    cv2.putText(npimg,"LHip",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==12:
                    cv2.putText(npimg,"LKnee",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==13:
                    cv2.putText(npimg,"LAnkle",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==14:
                    cv2.putText(npimg,"REye",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==15:
                    cv2.putText(npimg,"LEye",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==16:
                    cv2.putText(npimg,"REar",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                if i==17:
                    cv2.putText(npimg,"LEar",center,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
                '''

            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] == 1 and pair[1] == 8:
                    continue
                if pair[0] == 1 and pair[1] == 11:
                    continue
                if wrist_check == 1 and (pair[0] == 1 or pair[0] == 8 or pair[0] == 11):
                    npimg = npimg
                    npimg = cv2.line(npimg, centers[pair[0]], wrist_inference,(0,255,255), 3)
                    npimg = cv2.line(npimg, centers[pair[0]], wrist_inference, (rgb[j][0], rgb[j][1], rgb[j][2]), 3)
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys() or pair[0] > 13 or pair[1] > 13: # face remove
                    continue
                npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]],(0,255,255), 3)
                npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], (rgb[j][0], rgb[j][1], rgb[j][2]), 3)

        return npimg, checkPerson, trueHumanCount, joint_pixelX, joint_pixelY


    # temporary function for joint tracker  
    def joint_pointer(npimg, humans, imgcopy=False):
        if imgcopy:
            npimg = np.copy(npimg)
        image_h, image_w = npimg.shape[:2]
        joint_pixelX = [[1] * 19 for j in range(12)] # values setup
        joint_pixelY = [[1] * 19 for j in range(12)] # Because, None value -> error.
        centerX = 0
        centerY = 0
        multi_people = 0 # Same value of humans.

        checkPerson = np.ones(len(humans), dtype=bool)
        k = -1
        for human in humans:
            wrist_inference = (0,0)
            k+=1
            # Ignore if less than 10 joints
            if len(human.body_parts) < 10:
                checkPerson[k] = 0
                # print("continue")
                continue
            for i in range(common.CocoPart.Background.value): # all image data range

                if i not in human.body_parts.keys(): # body parts number
                    continue
                body_part = human.body_parts[i] # body parts number
                centerX = int(body_part.x * image_w + 0.5) # jointpoint X axis
                centerY = int(body_part.y * image_h + 0.5) # jointpoint Y axis
                joint_pixelX[multi_people][i] = centerX # saving X axis
                joint_pixelY[multi_people][i] = centerY # saving Y axis

                if i == 8:
                    wrist_inference = centerX,centerY

                if wrist_inference != (0,0) and i == 11:
                    joint_pixelX[multi_people][18],joint_pixelY[multi_people][18] = (int((wrist_inference[0] + centerX) / 2),int((wrist_inference[1] + centerY) / 2))
            multi_people += 1

        # print(checkPerson)

        # count found people
        # k = checkPerson.nonzero()
        # print(np.shape(k)[1])
        return joint_pixelX, joint_pixelY # return values

               
    def _get_scaled_img(self, npimg, scale):
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(w), self.target_size[1] / float(h)) * s
        img_h, img_w = npimg.shape[:2]

        if scale is None:
            if npimg.shape[:2] != (self.target_size[1], self.target_size[0]):
                # resize
                npimg = cv2.resize(npimg, self.target_size)
            return [npimg], [(0.0, 0.0, 1.0, 1.0)]
        elif isinstance(scale, float):
            # scaling with center crop
            base_scale = get_base_scale(scale, img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            ratio_x = (1. - self.target_size[0] / float(npimg.shape[1])) / 2.0
            ratio_y = (1. - self.target_size[1] / float(npimg.shape[0])) / 2.0
            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, 1.-ratio_x*2, 1.-ratio_y*2)]
        elif isinstance(scale, tuple) and len(scale) == 2:
            # scaling with sliding window : (scale, step)
            base_scale = get_base_scale(scale[0], img_w, img_h)
            base_scale_w = self.target_size[0] / (img_w * base_scale)
            base_scale_h = self.target_size[1] / (img_h * base_scale)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            window_step = scale[1]
            rois = []
            infos = []
            for ratio_x, ratio_y in itertools.product(np.arange(0., 1.01 - base_scale_w, window_step),
                                                      np.arange(0., 1.01 - base_scale_h, window_step)):
                roi = self._crop_roi(npimg, ratio_x, ratio_y)
                rois.append(roi)
                infos.append((ratio_x, ratio_y, base_scale_w, base_scale_h))
            return rois, infos
        elif isinstance(scale, tuple) and len(scale) == 3:
            # scaling with ROI : (want_x, want_y, scale_ratio)
            base_scale = get_base_scale(scale[2], img_w, img_h)
            npimg = cv2.resize(npimg, dsize=None, fx=base_scale, fy=base_scale)
            ratio_w = self.target_size[0] / float(npimg.shape[1])
            ratio_h = self.target_size[1] / float(npimg.shape[0])

            want_x, want_y = scale[:2]
            ratio_x = want_x - ratio_w / 2.
            ratio_y = want_y - ratio_h / 2.
            ratio_x = max(ratio_x, 0.0)
            ratio_y = max(ratio_y, 0.0)
            if ratio_x + ratio_w > 1.0:
                ratio_x = 1. - ratio_w
            if ratio_y + ratio_h > 1.0:
                ratio_y = 1. - ratio_h

            roi = self._crop_roi(npimg, ratio_x, ratio_y)
            return [roi], [(ratio_x, ratio_y, ratio_w, ratio_h)]

    def _crop_roi(self, npimg, ratio_x, ratio_y):
        target_w, target_h = self.target_size
        h, w = npimg.shape[:2]
        x = max(int(w*ratio_x-.5), 0)
        y = max(int(h*ratio_y-.5), 0)
        cropped = npimg[y:y+target_h, x:x+target_w]

        cropped_h, cropped_w = cropped.shape[:2]
        if cropped_w < target_w or cropped_h < target_h:
            npblank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)

            copy_x, copy_y = (target_w - cropped_w) // 2, (target_h - cropped_h) // 2
            npblank[copy_y:copy_y+cropped_h, copy_x:copy_x+cropped_w] = cropped
        else:
            return cropped

    def inference(self, npimg, scales=None):
        if not isinstance(scales, list):
            scales = [None]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)
            pass

        rois = []
        infos = []
        for scale in scales:
            roi, info = self._get_scaled_img(npimg, scale)
            # for dubug...
            # print(roi[0].shape)
            # cv2.imshow('a', roi[0])
            # cv2.waitKey()
            rois.extend(roi)
            infos.extend(info)

        logger.debug('inference+')
        output = self.persistent_sess.run(self.tensor_output, feed_dict={self.tensor_image: rois})
        heatMats = output[:, :, :, :19]
        pafMats = output[:, :, :, 19:]
        logger.debug('inference-')

        output_h, output_w = output.shape[1:3]
        max_ratio_w = max_ratio_h = 10000.0
        for info in infos:
            max_ratio_w = min(max_ratio_w, info[2])
            max_ratio_h = min(max_ratio_h, info[3])
        mat_w, mat_h = int(output_w/max_ratio_w), int(output_h/max_ratio_h)
        resized_heatMat = np.zeros((mat_h, mat_w, 19), dtype=np.float32)
        resized_pafMat = np.zeros((mat_h, mat_w, 38), dtype=np.float32)
        resized_cntMat = np.zeros((mat_h, mat_w, 1), dtype=np.float32)
        resized_cntMat += 1e-12

        for heatMat, pafMat, info in zip(heatMats, pafMats, infos):
            w, h = int(info[2]*mat_w), int(info[3]*mat_h)
            heatMat = cv2.resize(heatMat, (w, h))
            pafMat = cv2.resize(pafMat, (w, h))
            x, y = int(info[0] * mat_w), int(info[1] * mat_h)

            if TfPoseEstimator.ENSEMBLE == 'average':
                # average
                resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] += heatMat[max(0, -y):, max(0, -x):, :]
                resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                resized_cntMat[max(0,y):y+h, max(0, x):x+w, :] += 1
            else:
                # add up
                resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(resized_heatMat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
                resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                resized_cntMat[max(0, y):y + h, max(0, x):x + w, :] += 1

        if TfPoseEstimator.ENSEMBLE == 'average':
            self.heatMat = resized_heatMat / resized_cntMat
            self.pafMat = resized_pafMat / resized_cntMat
        else:
            self.heatMat = resized_heatMat
            self.pafMat = resized_pafMat / (np.log(resized_cntMat) + 1)

        humans = PoseEstimator.estimate(self.heatMat, self.pafMat)
        return humans
