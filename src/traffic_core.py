import cv2
import numpy as np
import math
from ultralytics import YOLO
import csv
import json
import os

# ============== PATHS ==============

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_video_path = os.path.join(BASE_DIR, "data", "traffic.mp4")
output_video_path = os.path.join(BASE_DIR, "results", "output_core.mp4")
events_csv_path = os.path.join(BASE_DIR, "results", "traffic_events_core.csv")
events_json_path = os.path.join(BASE_DIR, "results", "traffic_events_core.json")

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

# ============== MODEL & SETTINGS ==============

traffic_model = YOLO("rtdetr-l.pt")

traffic_classes = ["car", "truck", "bus", "motorcycle", "motorbike", "bicycle"]

meters_per_pixel = 0.1
line_position_ratio = 0.5 # vertical position of counting line

# ============== SIMPLE TRACKER ==============

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.tracks = {}
        self.max_distance = max_distance

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections):
        updated_tracks = {}
        used_ids = set()

        for det in detections:
            c_det = self._centroid(det["bbox"])
            best_id = None
            best_dist = 1e9

            for track_id, track in self.tracks.items():
                if track_id in used_ids:
                    continue
                c_tr = self._centroid(track["bbox"])
                dist = math.dist(c_det, c_tr)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_id = track_id

            if best_id is None:
                track_id = self.next_id
                self.next_id += 1
            else:
                track_id = best_id

            updated_tracks[track_id] = {
                "bbox": det["bbox"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "missed": 0
            }
            used_ids.add(track_id)

        for track_id, track in self.tracks.items():
            if track_id not in updated_tracks:
                track["missed"] += 1
                if track["missed"] <= 5:
                    updated_tracks[track_id] = track

        self.tracks = updated_tracks
        return self.tracks

# ============== TRAFFIC LIGHT COLOR (RED/GREEN) ==============

def detect_light_state(frame_rgb, light_boxes):
    if not light_boxes:
        return "UNKNOWN"

    x1, y1, x2, y2 = light_boxes[0]
    h, w, _ = frame_rgb.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2

    lower_green = np.array([35, 70, 50])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = np.count_nonzero(mask_red)
    green_pixels = np.count_nonzero(mask_green)

    if red_pixels > green_pixels * 1.5 and red_pixels > 20:
        return "RED"
    elif green_pixels > red_pixels * 1.5 and green_pixels > 20:
        return "GREEN"
    else:
        return "UNKNOWN"

# ============== MAIN PIPELINE ==============

def main():
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    line_y = int(h * line_position_ratio)

    tracker = SimpleTracker(max_distance=60)

    count_pass = 0
    prev_positions = {}
    track_history = {}
    last_speed_kmh = {}
    frame_idx = 0
    events = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ---- DETECTION ----
        results = traffic_model.predict(source=frame_rgb, verbose=False)[0]

        detections = []
        traffic_light_boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            class_name = traffic_model.names[cls_id]

            if class_name == "traffic light":
                traffic_light_boxes.append((x1, y1, x2, y2))
            elif class_name in traffic_classes:
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "confidence": conf
                })

        # ---- TRAFFIC LIGHT STATE ----
        light_state = detect_light_state(frame_rgb, traffic_light_boxes)

        # ---- TRACKING ----
        tracks = tracker.update(detections)

        annotated = frame_rgb.copy()

        # draw traffic light
        for (x1, y1, x2, y2) in traffic_light_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, "TL", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(
            annotated, f"Light: {light_state}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if light_state == "GREEN" else (0, 0, 255), 2
        )

        # counting line
        cv2.line(
            annotated, (0, line_y), (w, line_y),
            (0, 255, 0) if light_state != "RED" else (0, 0, 255), 2
        )

        # per track
        for track_id, tr in tracks.items():
            x1, y1, x2, y2 = map(int, tr["bbox"])
            class_name = tr["class_name"]
            conf = tr["confidence"]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((frame_idx, cx, cy))
            if len(track_history[track_id]) > 30:
                track_history[track_id] = track_history[track_id][-30:]

            # crossing
            if track_id in prev_positions:
                prev_cx, prev_cy = prev_positions[track_id]

                crossed = False
                speed_kmh = None
                event_type = None

                if (prev_cy < line_y and cy >= line_y) or (prev_cy > line_y and cy <= line_y):
                    crossed = True
                    count_pass += 1

                    hist = track_history[track_id]
                    if len(hist) > 5:
                        ref_frame_idx, ref_cx, ref_cy = hist[0]
                        pixel_dist = math.dist((ref_cx, ref_cy), (cx, cy))
                        time_sec = (frame_idx - ref_frame_idx) / fps
                        if time_sec > 0:
                            dist_m = pixel_dist * meters_per_pixel
                            speed_m_s = dist_m / time_sec
                            speed_kmh = speed_m_s * 3.6
                            last_speed_kmh[track_id] = speed_kmh

                    if light_state == "RED":
                        event_type = "RED_LIGHT_VIOLATION"
                    else:
                        event_type = "CROSSING"

                if crossed and event_type is not None:
                    events.append({
                        "frame": frame_idx,
                        "time_sec": frame_idx / fps,
                        "track_id": int(track_id),
                        "class_name": class_name,
                        "speed_kmh": float(speed_kmh) if speed_kmh is not None else None,
                        "event_type": event_type,
                        "light_state": light_state
                    })

            prev_positions[track_id] = (cx, cy)

            # draw boxes and text
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label1 = f"ID:{track_id} {class_name} {conf:.2f}"
            cv2.putText(annotated, label1, (x1, max(0, y1 - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.circle(annotated, (cx, cy), 3, (255, 0, 0), -1)

            if track_id in last_speed_kmh:
                speed_text = f"{last_speed_kmh[track_id]:.1f} km/h"
                cv2.putText(annotated, speed_text, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(
            annotated, f"CARS: {count_pass}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    print("Video finished:", output_video_path)

    # Save events
    if events:
        fieldnames = [
            "frame", "time_sec", "track_id", "class_name",
            "speed_kmh", "event_type", "light_state"
        ]

        with open(events_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in events:
                writer.writerow(e)
        print("Events saved to CSV:", events_csv_path)

        with open(events_json_path, "w") as f:
            json.dump(events, f, indent=2)
        print("Events saved to JSON:", events_json_path)
    else:
        print("No events. CSV/JSON not created.")


if __name__ == "__main__":
    main()
