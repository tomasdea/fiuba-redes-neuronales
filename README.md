# FIUBA - TP Final Redes Neuronales 86.54

La propuesta consiste en el desarrollo y análisis de un sistema basado en redes neuronales que, a partir de una secuencia corta de video (5 segundos), estime automáticamente una región de interés (ROI) donde probablemente haya presencia de personas. El objetivo es que este módulo actúe como una etapa previa para un sistema más complejo de análisis de violencia en video, permitiendo reducir el costo de cómputo: en vez de procesar el frame completo, la red “pesada” posterior procesaría únicamente el recorte indicado por la ROI (y, si no hay personas, devolvería una ROI nula).

En concreto, el sistema tomaría como entrada el video muestreado a baja tasa (por ejemplo 2 fps, de modo de obtener un tensor temporal acotado) y como salida un bounding box cuadrado (x1, y1, x2, y2) en coordenadas de píxeles, con un tamaño mínimo (p.ej. 224×224) y ubicación libre en el frame. Si no se detectan personas en ningún momento del clip, la salida sería (0, 0, 0, 0).

Para entrenar y evaluar el sistema, la idea es generar un dataset de clips con y sin personas y pseudo-etiquetar la ROI usando un detector de personas como “teacher” (por ejemplo una YOLO preentrenada) para construir un mapa de ocupación temporal y derivar la ROI objetivo, y luego entrenar un modelo de video (p.ej. una 3D-CNN o una arquitectura liviana con agregación temporal) que aprenda a predecir directamente la ROI desde el clip.

## Crear dataset
```bash
python build_video_roi_dataset_var.py \
  --videos_dir ~/Videos/roi_videos \
  --out_dir ~/Projects/redes_neuronales/tp_final/video_roi_dataset \
  --sample_fps 2 \
  --clip_seconds 5 \
  --resize_h 256 --resize_w 512 \
  --min_side 224 \
  --min_presence_ratio 0.02
```
## Entrenar 3D CNN que predice `present + (cx,cy) + side_norm`

```bash
python train_video_roi_var.py \
  --data_dir ~/Projects/redes_neuronales/tp_final/video_roi_dataset \
  --epochs 20 \
  --batch 4 \
  --lr 1e-3 \
  --out video_roi_var.pt
```

## Inferencia + JSON + overlay
```bash
python infer_video_roi_var_and_overlay.py \
  --video ~/Videos/security_0.mp4 \
  --weights video_roi_var.pt \
  --outdir pred_security0 \
  --sample_fps 2 \
  --clip_seconds 5 \
  --min_side 224
```