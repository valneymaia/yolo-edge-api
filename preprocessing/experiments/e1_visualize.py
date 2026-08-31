"""Gera comparativo visual BGR vs RGB para inspeção."""
import cv2
import numpy as np
from pathlib import Path


img_path = sorted(Path("dataset/exports/epi-v1/valid/images").glob("*.jpg"))[0]
frame    = cv2.imread(str(img_path))


bgr_display = frame.copy()                            # BGR — azul parece vermelho
rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # correto


# Salva os dois para comparação via SCP
cv2.imwrite("preprocessing/outputs/e1_rgb_correto.jpg", frame)
cv2.imwrite("preprocessing/outputs/e1_rgb_correto.jpg",
            cv2.cvtColor(rgb_display, cv2.COLOR_RGB2BGR))  # reconverte para salvar


print("Imagens salvas em preprocessing/outputs/")
print("Do seu computador, substitua 100.100.85.11 e rode:")
print("IP_DO_PI=100.100.85.11")
print("scp valneymaia@$100.100.85.11:~/yolo-edge-api/preprocessing/outputs/*.jpg .")
