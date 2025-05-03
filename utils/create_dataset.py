import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
from tqdm import tqdm

# Configuración optimizada
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=False,
    device=device,
    min_face_size=60,            # Reducido para INEs
    thresholds=[0.5, 0.6, 0.6],  # Umbrales más bajos
    post_process=False
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def adaptive_preprocess(img, is_ine=False):
    """Preprocesamiento adaptativo para fotos/INEs"""
    # Convertir a LAB y mejorar luminosidad
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE para INEs, normal para fotos
    clahe = cv2.createCLAHE(clipLimit=3.0 if is_ine else 1.5, 
                          tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Fusionar canales y convertir a RGB
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Enfoque especial para INEs
    if is_ine:
        # Reducción de ruido
        enhanced = cv2.medianBlur(enhanced, 3)
        # Aumentar contraste
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
    return enhanced

def extract_face(img_path, is_ine=False):
    """Extrae rostro con tolerancia a fallos"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Preprocesamiento adaptado al tipo de imagen
    img = adaptive_preprocess(img, is_ine)
    
    # Convertir a PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Intento 1: Detección estándar
    face = mtcnn(img_pil)
    
    # Intento 2: Si falla, escalar imagen
    if face is None:
        larger_img = img_pil.resize((img_pil.size[0]*2, img_pil.size[1]*2))
        face = mtcnn(larger_img)
    
    return face

def process_images():
    datasets = {
        'same': 1,
        'different': 0
    }
    
    data = []
    
    for folder, label in datasets.items():
        folder_path = os.path.join('datasets', folder)
        files = [f for f in os.listdir(folder_path) if '_foto' in f]
        
        for file in tqdm(files, desc=f'Procesando {folder}'):
            base = file.replace('_foto.jpg', '')
            foto_path = os.path.join(folder_path, file)
            ine_path = os.path.join(folder_path, f'{base}_ine.jpg')
            
            # Extraer embeddings
            face_foto = extract_face(foto_path)
            face_ine = extract_face(ine_path, is_ine=True)
            
            if face_foto is None or face_ine is None:
                continue
                
            with torch.no_grad():
                emb_foto = resnet(face_foto.unsqueeze(0).to(device)).cpu().numpy().flatten()
                emb_ine = resnet(face_ine.unsqueeze(0).to(device)).cpu().numpy().flatten()
            
            # Calcular similitud coseno
            sim = np.dot(emb_foto, emb_ine) / (np.linalg.norm(emb_foto) * np.linalg.norm(emb_ine))
            
            # Transformación para hacer más sensibles las diferencias altas
            similitud_transformada = np.power(sim, 2)  # o potencia 4, ajustable

            data.append({
                'persona': base,
                'label': label,
                'similitud': similitud_transformada,
                **{f'foto_{i}': val for i, val in enumerate(emb_foto)},
                **{f'ine_{i}': val for i, val in enumerate(emb_ine)}
            })
    
    return pd.DataFrame(data)

def start_process():
    df = process_images()
    df['similitud']
    os.makedirs('datasets', exist_ok=True)
    df.to_csv('datasets/embeddings.csv', index=False)
    print(f'\nCreacion del dataset completado. Muestras válidas: {len(df)}')