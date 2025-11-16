from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine

# --- Configuration ---
MODEL_NAME = "yolov8s.pt"
WEBCAM_ID = 0
TARGET_CLASS = 0  # person
CONFIDENCE_THRESH = 0.5
TRACKER_CONFIG_FILE = "memory.yaml"
REID_SIMILARITY_THRESH = 0.75  # Crește pentru mai multă precizie
MAX_FRAMES_MISSING = 300  # ~10 secunde la 30 FPS
FEATURE_HISTORY_SIZE = 5  # Păstrează ultimele 5 feature-uri


# ---------------------

class PersonReID:
    """Sistem de re-identificare robust pentru persoane"""

    def __init__(self, similarity_threshold=0.75, max_missing_frames=300):
        self.similarity_threshold = similarity_threshold
        self.max_missing_frames = max_missing_frames

        # {permanent_id: [feature1, feature2, ...]}
        self.person_features = {}

        # {tracker_id: permanent_id}
        self.tracker_to_permanent = {}

        # {permanent_id: frame_count}
        self.last_seen = {}

        # Contor ID-uri permanente
        self.next_id = 1
        self.frame_count = 0

    def extract_features(self, image, bbox):
        """Extrage histograme de culori + normalizare robustă"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64, 128))

        # Histograme HSV (mai robust la lumină decât BGR)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])  # Hue
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])  # Saturation
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()

        return np.concatenate([hist_h, hist_s])

    def compare(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        return 1 - cosine(f1, f2)

    def find_match(self, features):
        best_id = None
        best_sim = 0.0
        for pid, flist in self.person_features.items():
            sims = [self.compare(features, f) for f in flist]
            avg_sim = np.mean(sims)
            if avg_sim > best_sim and avg_sim >= self.similarity_threshold:
                best_sim = avg_sim
                best_id = pid
        return (best_id, best_sim) if best_id else (None, 0.0)

    def update(self, frame, detections):
        self.frame_count += 1
        current_trackers = set()

        for track_id, bbox in detections:
            current_trackers.add(track_id)
            features = self.extract_features(frame, bbox)
            if features is None:
                continue

            permanent_id = None
            if track_id in self.tracker_to_permanent:
                permanent_id = self.tracker_to_permanent[track_id]
            else:
                matched_id, sim = self.find_match(features)
                if matched_id:
                    permanent_id = matched_id
                    print(f"Re-ID: Tracker {track_id} → Persoană {permanent_id} (sim: {sim:.3f})")
                else:
                    permanent_id = self.next_id
                    self.next_id += 1
                    self.person_features[permanent_id] = []
                    print(f"Nouă persoană: ID {permanent_id}")

                self.tracker_to_permanent[track_id] = permanent_id

            # Actualizează feature-uri (ultimele N)
            self.person_features[permanent_id].append(features)
            if len(self.person_features[permanent_id]) > FEATURE_HISTORY_SIZE:
                self.person_features[permanent_id].pop(0)

            self.last_seen[permanent_id] = self.frame_count

        # Cleanup: elimină tracker-ele pierdute de mult
        to_remove = []
        for tid, pid in self.tracker_to_permanent.items():
            if tid not in current_trackers:
                if self.frame_count - self.last_seen[pid] > self.max_missing_frames:
                    to_remove.append(tid)

        for tid in to_remove:
            pid = self.tracker_to_permanent[tid]
            del self.tracker_to_permanent[tid]
            print(f"Eliminat tracker {tid} (ID {pid}) - absent {self.frame_count - self.last_seen[pid]} frame-uri")

        return self.tracker_to_permanent

    def cleanup_old_persons(self):
        """Elimină complet persoanele nevăzute de mult (opțional)"""
        to_remove = [pid for pid, last in self.last_seen.items()
                     if self.frame_count - last > self.max_missing_frames * 2]
        for pid in to_remove:
            if pid in self.person_features:
                del self.person_features[pid]
            if pid in self.last_seen:
                del self.last_seen[pid]
            print(f"Șters complet ID {pid} - nevăzut de mult")


# === Inițializare ===
reid = PersonReID(REID_SIMILARITY_THRESH, MAX_FRAMES_MISSING)
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(WEBCAM_ID)

if not cap.isOpened():
    print("Eroare: Camera nu poate fi deschisă.")
    exit()

print("ReID activat. Apasă 'q' pentru a ieși.")

# Culori pentru ID-uri
COLORS = np.random.randint(0, 255, (1000, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=TARGET_CLASS,
        conf=CONFIDENCE_THRESH,
        tracker=TRACKER_CONFIG_FILE
    )

    detections = []
    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, track_ids):
            detections.append((tid, box))

    # === Actualizează ReID ===
    id_map = reid.update(frame, detections)

    # === Desenează cu ID permanent ===
    if results[0].boxes.id is not None:
        for box, tid in zip(boxes, track_ids):
            if tid in id_map:
                pid = id_map[tid]
                x1, y1, x2, y2 = map(int, box)
                color = tuple(map(int, COLORS[pid % 1000]))

                # Fundal pentru text
                cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + 120, y1), color, -1)
                cv2.putText(annotated_frame, f"ID: {pid}", (x1 + 5, y1 - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

    # Statistici
    active = len(reid.person_features)
    total = reid.next_id - 1
    cv2.putText(annotated_frame, f"Active: {active} | Total: {total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + ReID Persistent", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTotal persoane unice detectate: {reid.next_id - 1}")