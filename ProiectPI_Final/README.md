# YOLOv8 Person detection and tracking system

Sistem de tracking și re-identificare persoane care menține ID-uri permanente chiar și când persoanele ies temporar din cadru sau sunt ocultate.

## Ce face?

Urmărește persoane în video și le asociază ID-uri permanente persistente. Când o persoană iese din cadru și revine (chiar și după multe frame-uri), sistemul o recunoaște automat după caracteristici vizuale și spațiale.


## Tehnologii folosite

- **YOLOv8** - Detecție obiecte în timp real
- **BoT-SORT** - Tracker multi-obiect 
- **OpenCV** - Procesare video și imagine
- **NumPy** - Calcule numerice și operații pe liste
- **SciPy** - Distanță cosinus pentru comparare feature-uri
- **Ultralytics** - Framework YOLO

## Instalare
```bash
pip install ultralytics opencv-python numpy scipy
```

## Fișiere necesare

- `Person_Tracking.py` - codul principal
- `yolov8s.pt` - model YOLOv8 pre-antrenat
- `memory.yaml` - fișier de configurare pentru tracker BoT-SORT

## Utilizare
```bash
python Person_Tracking.py
```

Apasă `q` sau închide fereastra pentru a opri aplicația.

## Configurare

### Parametri principali în cod:
```python
MODEL_NAME = r"yolov8s.pt"                    # Modelul YOLO
WEBCAM_ID = 0                                 # 0 pentru webcam, sau path la video
CONFIDENCE_THRESH = 0.5                       # Prag de încredere detecție (0.0-1.0)
REID_SIMILARITY_THRESH = 0.75                 # Prag similaritate Re-ID (0.0-1.0)
MAX_FRAMES_MISSING = 1000000                  # Frame-uri maxime pentru re-identificare
FEATURE_HISTORY_SIZE = 10                     # Istoric feature-uri per persoană
IOU_THRESHOLD = 0.3                           # Overlap minim între bounding boxes
WINDOW_WIDTH = 1280                           # Lățime fereastră afișare
WINDOW_HEIGHT = 720                           # Înălțime fereastră afișare
```

### Pentru a folosi un video:
```python
cap = cv2.VideoCapture(r"path\to\video.mp4")
```

### Pentru a folosi webcam:
```python
cap = cv2.VideoCapture(0)  
```

## Cum funcționează

### Arhitectură sistem

1. **Detecție YOLOv8** - Identifică persoane în fiecare frame cu YOLOv8 + BoT-SORT tracker
2. **Extracție caracteristici vizuale** - Salvează histograme HSV, proporții și pattern-uri pentru fiecare persoană
3. **Sistem de matching multi-criteriu** - Compară detecții noi cu baza de date folosind:
   - Similaritate vizuală (distanță cosinus între feature-uri)
   - Proximitate spațială (IoU între bounding boxes)
   - Continuitate temporală (preferință pentru persoane văzute recent)
4. **Re-identificare** - Asociază automat ID-uri permanente chiar după ocultări prelungite
5. **Corecție automată** - Detectează și corectează asocieri greșite bazate pe scor de încredere

### Extracție caracteristici (Features)

Sistemul extrage multiple tipuri de caracteristici pentru fiecare persoană:

- **Histograme HSV globale** (50+32+32 bins) - captează culoarea generală a îmbrăcămintei
- **Histograme segmentate top/bottom** (32x32 bins fiecare) - diferențiază tricou vs pantaloni
- **Caracteristici geometrice** - lățime relativă, înălțime relativă, aspect ratio

### Algoritm de matching

Pentru fiecare detecție nouă, sistemul:

1. Filtrează candidații invalizi (ID-uri asociate cu alți trackeri activi, prea vechi)
2. Calculează scoruri multiple:
   - **Scor de similaritate** - media și max dintre toate feature-urile stocate
   - **Scor spațial** - IoU cu ultima poziție cunoscută
   - **Scor temporal** - decădere exponențială în funcție de frame-uri absente
3. Combină scorurile: `0.5×avg_sim + 0.2×max_sim + 0.2×spatial + 0.1×temporal`
4. Selectează cel mai bun candidat dacă similaritatea > 60% din threshold (0.75)
5. Verifică ambiguitate - dacă 2 candidați au scoruri apropiate, necesită similaritate mai mare

## Interfața grafică

### Elemente afișate:

- **Bounding boxes colorate** - fiecare persoană are o culoare consistentă
- **ID permanent** - afișat deasupra fiecărei persoane
- **Intensitate culoare** - transparența boxului reflectă încrederea în ID
- **Bară de statistici**:
  - **Active** - Persoane vizibile în frame-ul curent
  - **Total IDs** - Număr total de persoane unice detectate
  - **Trackers** - Număr de tracker-e active BoT-SORT


### Metrici evaluate:

- **Persoane unice** - Total ID-uri permanente create
- **Factor de încredere vizual** - Similaritate medie între frame-uri consecutive (stabilitate tracking)
- **mIoU** - Mean Intersection over Union (consistența pozițiilor spațiale)
- **Re-ID score** - Similaritatea medie la re-identificări (calitatea recuperării după ocultări)

