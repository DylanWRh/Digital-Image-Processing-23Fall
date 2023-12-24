import cv2
import numpy as np
import imutils.feature.factories as kp_factory

def get_transform_by_flow(optical_flow, prev_kps):
    cur_kps, status, _ = optical_flow
    
    cur_matched = cur_kps[status==1]
    prev_matched = prev_kps[status==1]
    
    transform = cv2.estimateAffinePartial2D(prev_matched, cur_matched)[0]
    
    if transform is not None:
        dx = transform[0, 2]
        dy = transform[1, 2]
        da = np.arctan2(transform[1, 0], transform[0, 0])
    else:
        dx = dy = da = 0
    return [dx, dy, da]


def transform_to_matrix(transform):
    transform_matrix = np.zeros((2, 3))

    transform_matrix[0, 0] = np.cos(transform[2])
    transform_matrix[0, 1] = -np.sin(transform[2])
    transform_matrix[1, 0] = np.sin(transform[2])
    transform_matrix[1, 1] = np.cos(transform[2])
    transform_matrix[0, 2] = transform[0]
    transform_matrix[1, 2] = transform[1]

    return transform_matrix

class Video:
    def __init__(self, input_path: str):
        self.source = cv2.VideoCapture(input_path)
        self.fps = int(self.source.get(cv2.CAP_PROP_FPS))
        self.shape = (int(self.source.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(self.source.get(cv2.CAP_PROP_FRAME_WIDTH)))
    
    def read_frame(self):
        return self.source.read()

    def reset(self):
        self.source.set(cv2.CAP_PROP_POS_FRAMES, 0)

class VideoStabilizer:
    def __init__(self, kp_detector = None):
        self.kp_detector = kp_detector if kp_detector is not None else \
            kp_factory.FeatureDetector_create(
                'GFTT', maxCorners=200, qualityLevel=0.01, minDistance=30.0, blockSize=3
            )

        self.raw_transforms = None 
        self.trajectory = None
        self.smoothed_trajectory = None
        self.transforms = None

        self.video = None 
        self.writer = None 
    
    def init_writer(self, output_path: str, fourcc: str = 'MJPG'):
        assert self.video is not None

        h, w = self.video.shape
        fps = self.video.fps 

        self.writer = cv2.VideoWriter(output_path, 
                                      cv2.VideoWriter_fourcc(*fourcc), 
                                      fps, (w, h), True)
    
    def stabilize(self, 
                  smoothing_window: int = 30,
                  input_path: str = 'demo/demo.mp4', 
                  output_path: str = 'demo/output.avi'):
        self.video = Video(input_path)
        if output_path is not None:
            self.init_writer(output_path)
        else:
            self.writer = None

        # Step 1: Get transformation from previous to current frame
        raw_transforms = [[0, 0, 0]]
        isvalid, frame = self.video.read_frame()
        while isvalid:
            # Extract previous frame key points
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_kps = self.kp_detector.detect(prev_gray)
            prev_kps = np.array([kp.pt for kp in prev_kps], dtype=np.float32).reshape(-1, 1, 2)

            # Get current frame
            isvalid, frame = self.video.read_frame()
            if not isvalid:
                break
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate transformation
            optical_flow = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_kps, None)
            raw_transform = get_transform_by_flow(optical_flow, prev_kps)
            raw_transforms.append(raw_transform)
        self.raw_transforms = np.array(raw_transforms)

        # Step 2: Accumulate transforms to get image trajectory
        self.trajectory = np.cumsum(self.raw_transforms, axis=0)
        
        # Step 3: Smooth the trahectory with size of smoothing_window
        smoothed_trajectory = []
        for i in range(len(self.trajectory)):
            slice_min = max(i - smoothing_window, 0)
            slice_max = min(i + smoothing_window, len(self.trajectory))
            smoothed_trajectory.append(
                self.trajectory[slice_min:slice_max].mean(axis=0)
            )
        self.smoothed_trajectory = np.array(smoothed_trajectory)
        
        # Step 4: Generate new transformation from trajecroty
        diff_trans = self.smoothed_trajectory - self.trajectory
        self.transforms = self.raw_transforms + diff_trans
        
        # Step 5: Apply new transformation to video
        self.video.reset()
        for t in self.transforms:
            isvalid, frame = self.video.read_frame()
            if not isvalid:
                break

            transform_matrix = transform_to_matrix(t)
            h, w = self.video.shape
            transformed_frame = cv2.warpAffine(
                frame, transform_matrix, (w, h), 
                borderMode=cv2.BORDER_CONSTANT
            )

            if self.writer is not None:
                self.writer.write(transformed_frame)
        
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        cv2.destroyAllWindows()
        # self.__init__()

