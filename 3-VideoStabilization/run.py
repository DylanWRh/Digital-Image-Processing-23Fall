import argparse
from VideoStabilizer import VideoStabilizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoothing_window', type=int, default=30)
    parser.add_argument('--input_path', type=str, default='demo/demo.mp4')
    parser.add_argument('--output_path', type=str, default='demo/output.avi')
    args = parser.parse_args()

    VideoStabilizer().stabilize(args.smoothing_window, args.input_path, args.output_path)