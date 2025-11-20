# YOLOv8 Person Re-Identification

Sistem simplu de tracking persoane care menține același ID chiar dacă persoana iese temporar din cadru.

## Ce face?

Urmărește persoane în video și le asociază ID-uri permanente. Când o persoană iese din cadru și revine, sistemul o recunoaște după îmbrăcăminte și caracteristici vizuale.

## Instalare
```bash
pip install ultralytics opencv-python numpy scipy
```

## Fișiere necesare

- `Person_Tracking.py` - codul principal
- `yolov8s.pt` - model YOLOv8 (se descarcă automat)
- `memory.yaml` - configurare tracker

## Utilizare
```bash
python Person_Tracking.py
```

Apasă `q` pentru a opri.

## Configurare

Modifică în cod pentru a folosi un video în loc de webcam:
```python
WEBCAM_ID = "video.mp4"  # în loc de 0
```

## Parametri importanți
```python
CONFIDENCE_THRESH = 0.5           # Cât de sigure trebuie să fie detecțiile
REID_SIMILARITY_THRESH = 0.75     # Cât de similare pentru re-identificare
FEATURE_HISTORY_SIZE = 10         # Câte "memorii" păstrăm per persoană
```

## Cum funcționează

1. **Detecție** - YOLOv8 găsește persoane în fiecare frame
2. **Extracție features** - Salvează culori și caracteristici vizuale
3. **Comparare** - Când apare cineva nou, compară cu persoanele cunoscute
4. **Re-ID** - Dacă e similar cu cineva văzut înainte, folosește același ID

## Afișare

- **ID:** - ID-ul permanent al persoanei
- **Active** - Câte persoane sunt în cadru acum
- **Total IDs** - Câte persoane diferite au fost detectate
- **Trackers** - Câte tracker-e active

## Features folosite

- Histograme culoare HSV (pentru a recunoaște îmbrăcămintea)
- Analiză top/bottom (tricou vs pantaloni)
- Proporții bounding box

## Note

- Funcționează mai bine cu îmbrăcăminte colorată/distinctă
- Luminozitatea și unghiul camerei afectează performanța
- Pentru mai multă acuratețe, folosește YOLOv8m sau YOLOv8l în loc de yolov8s