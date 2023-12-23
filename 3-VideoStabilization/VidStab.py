"""VidStab: a class for stabilizing video files"""

import os
import time
import warnings
import cv2
import numpy as np
import imutils
import imutils.feature.factories as kp_factory
import matplotlib.pyplot as plt
import vidstab_utils
from frame_queue import FrameQueue
from frame import Frame

def bfill_rolling_mean(arr, n=30):
    """Helper to perform trajectory smoothing

    :param arr: Numpy array of frame trajectory to be smoothed
    :param n: window size for rolling mean
    :return: smoothed input arr

    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> bfill_rolling_mean(arr, n=2)
    array([[2.5, 3.5, 4.5],
           [2.5, 3.5, 4.5]])
    """
    if arr.shape[0] < n:
        raise ValueError('arr.shape[0] cannot be less than n')
    if n == 1:
        return arr

    pre_buffer = np.zeros(3).reshape(1, 3)
    post_buffer = np.zeros(3 * n).reshape(n, 3)
    arr_cumsum = np.cumsum(np.vstack((pre_buffer, arr, post_buffer)), axis=0)
    buffer_roll_mean = (arr_cumsum[n:, :] - arr_cumsum[:-n, :]) / float(n)
    trunc_roll_mean = buffer_roll_mean[:-n, ]

    bfill_size = arr.shape[0] - trunc_roll_mean.shape[0]
    bfill = np.tile(trunc_roll_mean[0, :], (bfill_size, 1))

    return np.vstack((bfill, trunc_roll_mean))


class VidStab:
    def __init__(self,  *args, **kwargs):

        self.kp_detector = kp_factory.FeatureDetector_create('GFTT',
                                                                 maxCorners=200,
                                                                 qualityLevel=0.01,
                                                                 minDistance=30.0,
                                                                 blockSize=3)
        self._smoothing_window = 30
        self._raw_transforms = []
        self._trajectory = []
        self.trajectory = self.smoothed_trajectory = self.transforms = None

        self.frame_queue = FrameQueue()
        self.prev_kps = self.prev_gray = None

        self.writer = None

        self.border_options = {}

    def _update_prev_frame(self, current_frame_gray):
        self.prev_gray = current_frame_gray[:]
        self.prev_kps = self.kp_detector.detect(self.prev_gray)
        # noinspection PyArgumentList
        self.prev_kps = np.array([kp.pt for kp in self.prev_kps], dtype='float32').reshape(-1, 1, 2)

    def _update_trajectory(self, transform):
        if not self._trajectory:
            self._trajectory.append(transform[:])
        else:
            # gen cumsum for new row and append
            self._trajectory.append([self._trajectory[-1][j] + x for j, x in enumerate(transform)])

    def _gen_next_raw_transform(self):
        current_frame = self.frame_queue.frames[-1]
        current_frame_gray = current_frame.gray_image

        # calc flow of movement
        optical_flow = cv2.calcOpticalFlowPyrLK(self.prev_gray,
                                                current_frame_gray,
                                                self.prev_kps, None)

        matched_keypoints = vidstab_utils.match_keypoints(optical_flow, self.prev_kps)
        transform_i = vidstab_utils.estimate_partial_transform(matched_keypoints)

        # update previous frame info for next iteration
        self._update_prev_frame(current_frame_gray)
        self._raw_transforms.append(transform_i[:])
        self._update_trajectory(transform_i)

    def _init_is_complete(self):

        max_ind = min([self.frame_queue.max_frames,
                       self.frame_queue.max_len])

        if self.frame_queue.inds[-1] >= max_ind - 1:
            return True

        return False

    def _process_first_frame(self):
        # read first frame
        _, _, _ = self.frame_queue.read_frame(pop_ind=False)

        # convert to gray scale
        prev_frame = self.frame_queue.frames[-1]
        prev_frame_gray = prev_frame.gray_image

        # detect keypoints
        prev_kps = self.kp_detector.detect(prev_frame_gray)
        # noinspection PyArgumentList
        self.prev_kps = np.array([kp.pt for kp in prev_kps], dtype='float32').reshape(-1, 1, 2)

        self.prev_gray = prev_frame_gray[:]

    def _init_trajectory(self, smoothing_window):
        self._smoothing_window = smoothing_window

        self._process_first_frame()


        # iterate through frames
        while True:
            # read current frame
            _, _, break_flag = self.frame_queue.read_frame(pop_ind=False)

            self._gen_next_raw_transform()

            if self._init_is_complete():
                break


        self._gen_transforms()

        return None

    def _init_writer(self, output_path, frame_shape, output_fourcc, fps):
        # set output and working dims
        h, w = frame_shape

        # setup video writer
        self.writer = cv2.VideoWriter(output_path,
                                      cv2.VideoWriter_fourcc(*output_fourcc),
                                      fps, (w, h), True)

    def _set_border_options(self, border_size, border_type):

        self.border_options = {
            'border_type': border_type,
            'border_size': border_size,
        }        

    def _gen_transforms(self):
        self.trajectory = np.array(self._trajectory)
        self.smoothed_trajectory = bfill_rolling_mean(self.trajectory, n=self._smoothing_window)
        self.transforms = np.array(self._raw_transforms) + (self.smoothed_trajectory - self.trajectory)

        # Dump superfluous frames
        # noinspection PyProtectedMember
        n = self.frame_queue._max_frames
        if n:
            self.trajectory = self.trajectory[:n - 1, :]
            self.smoothed_trajectory = self.smoothed_trajectory[:n - 1, :]
            self.transforms = self.transforms[:n - 1, :]


    def _apply_next_transform(self, i, frame_i):
        self._gen_transforms()

        if i is None:
            i = self.frame_queue.inds.popleft()

        if frame_i is None:
            frame_i = self.frame_queue.frames.popleft()

        try:
            transform_i = self.transforms[i, :]
        except IndexError:
            return None

        transformed = vidstab_utils.transform_frame(frame_i,
                                                    transform_i,
                                                    self.border_options['border_size'],
                                                    self.border_options['border_type'])
        transformed = transformed.cvt_color(frame_i.color_format)

        return transformed

    def stabilize(self, input_path, output_path, smoothing_window=30,
                  border_type='black', border_size=0, 
                  output_fourcc='MJPG',):
        self.writer = None

        if not os.path.exists(input_path) and not isinstance(input_path, int):
            raise FileNotFoundError(f'{input_path} does not exist')

        self.frame_queue.set_frame_source(cv2.VideoCapture(input_path))

        self.frame_queue.reset_queue(max_len=smoothing_window + 1, max_frames=float('inf'))

        self._init_trajectory(smoothing_window)

        self._set_border_options(border_size, border_type)
        while True:
            i, frame_i, break_flag = self.frame_queue.read_frame()
            if not self.frame_queue.frames_to_process() or break_flag:
                break

            self._gen_next_raw_transform()
            transformed = self._apply_next_transform(i, frame_i)
            fps_force = self.frame_queue.source_fps

            if self.writer is None:
                self._init_writer(output_path, transformed.shape[:2], output_fourcc,
                                  fps=fps_force)

            self.writer.write(transformed)

        self.writer.release()
        self.writer = None
        cv2.destroyAllWindows()

