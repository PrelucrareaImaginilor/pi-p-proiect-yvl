from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from collections import deque

# --- Configuration ---
MODEL_NAME = "pi-p-proiect-yvl\ProiectPI\yolov8s.pt"
WEBCAM_ID = 0
TARGET_CLASS = 0  # detectăm doar persoane
CONFIDENCE_THRESH = 0.5
TRACKER_CONFIG_FILE = "pi-p-proiect-yvl\ProiectPI\memory.yaml"
REID_SIMILARITY_THRESH = 0.75  # cât de similare trebuie să fie feature-urile pentru match
MAX_FRAMES_MISSING = 1000000  # permite re-identificarea chiar și după multe frame-uri
FEATURE_HISTORY_SIZE = 10  # câte seturi de feature-uri păstrăm per persoană
IOU_THRESHOLD = 0.3  # overlap minim între bounding boxes
WINDOW_WIDTH = 1280  # lățimea ferestrei de afișare
WINDOW_HEIGHT = 720  # înălțimea ferestrei de afișare


class PersonTracking:
    """
    Sistem de re-identificare a persoanelor bazat pe caracteristici vizuale.
    Menține ID-uri permanente chiar dacă trackingul se pierde temporar.
    """

    def __init__(self, similarity_threshold=0.75, max_missing_frames=300):
        self.similarity_threshold = similarity_threshold
        self.max_missing_frames = max_missing_frames

        # Stocare feature-uri pentru fiecare persoană identificată
        self.person_features = {}  # {permanent_id: deque([features])}

        # Mapare între ID-uri temporare (de la tracker) și ID-uri permanente
        self.tracker_to_permanent = {}  # {tracker_id: permanent_id}

        # Info despre ultima apariție a fiecărei persoane
        self.last_seen = {}  # {permanent_id: frame_count}
        self.last_bbox = {}  # {permanent_id: bbox}

        # Scor de încredere pentru fiecare ID
        self.id_confidence = {}  # {permanent_id: confidence_score}

        # Management ID-uri
        self.next_id = 1
        self.frame_count = 0

        self.assignment_history = {}

    def extract_features(self, image, bbox):
        """
        Extrage caracteristici vizuale din imagine pentru o persoană.
        Folosește histograme de culoare HSV și proporții bbox.
        """
        # Validare și crop la bounding box
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
            return None

        roi = image[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64, 128))  # dimensiune standard

        # 1. Histograme HSV globale (pentru culori generale)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])

        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        # 2. Histograme separate pentru partea de sus și de jos
        # (útil pentru diferențierea îmbrăcămintei - tricou vs pantaloni)
        h_roi = roi.shape[0]
        top_half = roi[:h_roi // 2, :]
        bottom_half = roi[h_roi // 2:, :]

        hsv_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
        hsv_bottom = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)

        hist_top = cv2.calcHist([hsv_top], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist_bottom = cv2.calcHist([hsv_bottom], [0, 1], None, [32, 32], [0, 180, 0, 256])

        hist_top = cv2.normalize(hist_top, hist_top).flatten()
        hist_bottom = cv2.normalize(hist_bottom, hist_bottom).flatten()

        # 3. Caracteristici geometrice (proporții bbox)
        bbox_features = np.array([
            (x2 - x1) / w,  # lățime relativă
            (y2 - y1) / h,  # înălțime relativă
            (y2 - y1) / (x2 - x1 + 1e-6)  # aspect ratio
        ])

        # Combinăm toate caracteristicile cu ponderi diferite
        features = np.concatenate([
            hist_h * 0.3,
            hist_s * 0.2,
            hist_v * 0.1,
            hist_top * 0.2,
            hist_bottom * 0.2,
            bbox_features * 0.1
        ])

        return features

    def calculate_iou(self, bbox1, bbox2):
        """Calculează Intersection over Union între două bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Coordonate intersecție
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Arie totală
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def compare(self, f1, f2):
        """Compară similaritatea între două seturi de feature-uri (1 = identic, 0 = diferit)"""
        if f1 is None or f2 is None:
            return 0.0
        return 1 - cosine(f1, f2)

    def find_match(self, features, bbox, current_tracker_id):
        """
        Caută cel mai bun match pentru feature-urile curente printre persoanele cunoscute.
        Ia în considerare: similaritate feature-uri, proximitate spațială, continuitate temporală.
        """
        candidates = []

        for pid, flist in self.person_features.items():
            # Nu lua în considerare ID-uri care sunt deja asociate cu alt tracker activ
            if pid in self.tracker_to_permanent.values():
                active_tracker = [t for t, p in self.tracker_to_permanent.items() if p == pid]
                if active_tracker and active_tracker[0] != current_tracker_id:
                    continue

            # Verifică cât timp a trecut de la ultima vedere
            frames_missing = self.frame_count - self.last_seen.get(pid, 0)

            if frames_missing > self.max_missing_frames:
                continue

            # Calculează similaritatea cu toate feature-urile stocate
            sims = [self.compare(features, f) for f in flist]
            avg_sim = np.mean(sims)
            max_sim = np.max(sims)

            # Verifică proximitatea spațială (dacă persoana e în zona așteptată)
            spatial_score = 0.0
            if pid in self.last_bbox:
                iou = self.calculate_iou(bbox, self.last_bbox[pid])
                spatial_score = iou

            # Scor temporal (preferă persoane văzute recent)
            temporal_score = 1.0 / (1.0 + frames_missing / 30.0)

            # Scor combinat cu ponderi
            combined_score = (
                    avg_sim * 0.5 +
                    max_sim * 0.2 +
                    spatial_score * 0.2 +
                    temporal_score * 0.1
            )

            if avg_sim >= self.similarity_threshold * 0.8:
                candidates.append((pid, combined_score, avg_sim, spatial_score))

        if not candidates:
            return None, 0.0

        # Găsește cel mai bun candidat
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_score, best_sim, best_spatial = candidates[0]

        # Verifică dacă există ambiguitate (mai multe match-uri similare)
        if len(candidates) > 1:
            second_score = candidates[1][1]
            if best_score - second_score < 0.1:
                print(f"  Ambiguitate: {best_pid} ({best_score:.3f}) vs {candidates[1][0]} ({second_score:.3f})")
                if best_sim < self.similarity_threshold:
                    return None, 0.0

        return best_pid, best_score

    def update(self, frame, detections):
        """
        Actualizează sistemul cu detecțiile din frame-ul curent.
        Asociază tracker ID-uri temporare cu ID-uri permanente.
        """
        self.frame_count += 1
        current_trackers = set()

        # Sortează detecțiile după mărime (bbox mai mari = mai fiabile)
        detections_sorted = sorted(
            detections,
            key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]),
            reverse=True
        )

        for track_id, bbox in detections_sorted:
            current_trackers.add(track_id)
            features = self.extract_features(frame, bbox)

            if features is None:
                continue

            permanent_id = None

            # Verifică dacă tracker-ul are deja un ID permanent asociat
            if track_id in self.tracker_to_permanent:
                permanent_id = self.tracker_to_permanent[track_id]

                # Verificare consistență (dacă a fost absent mult timp)
                if self.frame_count - self.last_seen.get(permanent_id, 0) > 60:
                    matched_id, score = self.find_match(features, bbox, track_id)
                    if matched_id and matched_id != permanent_id and score > 0.85:
                        print(f"Corecție: Tracker {track_id} re-asociat de la ID {permanent_id} → {matched_id}")
                        permanent_id = matched_id
                        self.tracker_to_permanent[track_id] = permanent_id
            else:
                # Încearcă să găsească un match cu persoane existente
                matched_id, score = self.find_match(features, bbox, track_id)

                if matched_id:
                    permanent_id = matched_id
                    print(f"Re-ID: Tracker {track_id} → Persoană {permanent_id} (scor: {score:.3f})")
                else:
                    # Persoană nouă - creează ID nou
                    permanent_id = self.next_id
                    self.next_id += 1
                    self.person_features[permanent_id] = deque(maxlen=FEATURE_HISTORY_SIZE)
                    self.id_confidence[permanent_id] = 0.5
                    print(f"Persoană nouă: ID {permanent_id}")

                self.tracker_to_permanent[track_id] = permanent_id

            # Actualizează baza de date de feature-uri
            if permanent_id not in self.person_features:
                self.person_features[permanent_id] = deque(maxlen=FEATURE_HISTORY_SIZE)

            self.person_features[permanent_id].append(features)

            # Actualizează informații temporale și spațiale
            self.last_seen[permanent_id] = self.frame_count
            self.last_bbox[permanent_id] = bbox

            # Crește confidence-ul pe măsură ce vedem persoana mai des
            if permanent_id in self.id_confidence:
                self.id_confidence[permanent_id] = min(1.0, self.id_confidence[permanent_id] + 0.05)

        # Cleanup: elimină trackere inactive
        inactive_trackers = set(self.tracker_to_permanent.keys()) - current_trackers
        for tid in inactive_trackers:
            pid = self.tracker_to_permanent[tid]
            # Păstrăm ID-ul permanent și feature-urile pentru re-identificare ulterioară
            del self.tracker_to_permanent[tid]
            print(f"Eliberat tracker {tid} (ID {pid})")

        return self.tracker_to_permanent

    def get_stats(self):
        """Returnează statistici despre tracking"""
        active_ids = set(self.tracker_to_permanent.values())
        total_ids = self.next_id - 1

        return {
            'active': len(active_ids),
            'total': total_ids,
            'trackers': len(self.tracker_to_permanent)
        }


# === Inițializare sistem ===
reid = PersonTracking(REID_SIMILARITY_THRESH, MAX_FRAMES_MISSING)
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(WEBCAM_ID)

if not cap.isOpened():
    print("Eroare: Video nu poate fi deschis.")
    exit()

print("ReID îmbunătățit activat. Apasă 'q' pentru ieșire.")

# Generare culori consistente pentru vizualizare
np.random.seed(42)
COLORS = np.random.randint(0, 255, (1000, 3))

frame_count = 0

# === Loop principal ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Redimensionare frame la dimensiunea ferestrei (1280x720)
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)

    # Rulează YOLO cu tracking
    results = model.track(
        frame,
        persist=True,
        classes=TARGET_CLASS,
        conf=CONFIDENCE_THRESH,
        tracker=TRACKER_CONFIG_FILE,
        verbose=False
    )

    # Extrage detecții
    detections = []
    annotated_frame = frame.copy()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, tid in zip(boxes, track_ids):
            detections.append((tid, box))

    # Actualizează sistemul ReID
    id_map = reid.update(frame, detections)

    # Desenează bounding boxes și ID-uri
    if results[0].boxes.id is not None:
        for box, tid in zip(boxes, track_ids):
            if tid in id_map:
                pid = id_map[tid]
                x1, y1, x2, y2 = map(int, box)
                color = tuple(map(int, COLORS[pid % 1000]))

                # Intensitatea culorii depinde de confidence
                conf = reid.id_confidence.get(pid, 0.5)
                alpha = int(conf * 200) + 55

                # Desenează box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                # Label cu background
                label = f"ID:{pid}"
                (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - h_label - 10),
                            (x1 + w_label + 10, y1), color, -1)

                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # Afișează statistici
    stats = reid.get_stats()
    info_bg = np.zeros((80, annotated_frame.shape[1], 3), dtype=np.uint8)
    info_bg[:] = (40, 40, 40)

    cv2.putText(info_bg, f"Active: {stats['active']} | Total IDs: {stats['total']} | Trackers: {stats['trackers']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(info_bg, f"Frame: {frame_count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    final_frame = np.vstack([info_bg, annotated_frame])
    cv2.imshow("YOLOv8 + ReID Multi-Person", final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or cv2.getWindowProperty("YOLOv8 + ReID Multi-Person", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

stats = reid.get_stats()