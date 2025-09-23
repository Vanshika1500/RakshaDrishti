# run_pose_with_logger.py
import time, json, os, argparse
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=0, help='Webcam index or video file')
    p.add_argument('--imgsz', type=int, default=320)
    p.add_argument('--log', default='Data/keypoints_logs.jsonl', help='File to save keypoints')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--show', action='store_true', help='Show preview window')
    p.add_argument('--label',default = None, help='Label for this recording.')
    return p.parse_args()

def norm_kps(kps, frame_w, frame_h):
    """Normalize YOLOv8 keypoints safely for any version"""
    norm = []
    for joint in kps:
        # joint can be list or array
        try:
            if len(joint) == 3:
                x, y, c = joint
            elif len(joint) == 2:
                x, y = joint
                c = 1.0
            else:
                # unexpected format
                x = y = c = 0.0
        except TypeError:
            # if joint is a single number or None
            x = y = c = 0.0

        # handle NaN / None
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            x = y = c = 0.0

        norm.append([float(x)/frame_w, float(y)/frame_h, float(c)])

    return norm



def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        raise SystemExit("Cannot open source")

    model = YOLO('yolov8n-pose.pt')
    frame_idx = 0

    with open(args.log, 'a') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            H, W = frame.shape[:2]

            results = model(frame, imgsz=args.imgsz, conf=args.conf)[0]

            persons = []
            try:
                kps_all = results.keypoints.cpu().numpy()
            except Exception:
                kps_all = None
            try:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
            except Exception:
                boxes = None
                confs = None

            if kps_all is not None:
                for i, kp in enumerate(kps_all):
                    norm = norm_kps(kp, W, H)
                    bbox = boxes[i].tolist() if boxes is not None else None
                    conf = float(confs[i]) if confs is not None else None
                    persons.append({'bbox': bbox, 'keypoints': norm, 'box_conf': conf})

            log = {
                't': time.time(),
                'frame_idx': frame_idx,
                'label': args.label,
                'persons': persons,
                'frame_size': [W,H]
            }
            f.write(json.dumps(log) + '\n')

            if args.show:
                vis = results.plot()
                cv2.imshow("pose_preview", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
