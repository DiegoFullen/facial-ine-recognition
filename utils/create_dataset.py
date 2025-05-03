import os
import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_curve, auc
from facenet_pytorch import MTCNN, InceptionResnetV1

# Configuración optimizada
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Configuración mejorada de MTCNN
mtcnn = MTCNN(
    keep_all=True,  # Detectar todas las caras y seleccionar la mejor
    device=device,
    min_face_size=50,  # Aún más reducido para INEs de baja calidad
    thresholds=[0.6, 0.7, 0.7],  # Umbrales ajustados
    post_process=True  # Activar post-procesamiento
)

# Modelo de features
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def enhance_image(img):
    """Mejora la calidad de la imagen para mejor detección"""
    # Convertir a LAB para mejorar la luminosidad
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE (mejora de contraste adaptativo)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Fusionar canales y convertir a RGB
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Reducción de ruido
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return enhanced

def adaptive_preprocess(img, is_ine=False):
    """Preprocesamiento adaptativo mejorado"""
    # Conversión inicial
    if len(img.shape) == 2:  # Es escala de grises
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Mejora básica
    enhanced = enhance_image(img)
    
    # Procesamiento específico para INEs
    if is_ine:
        # Aumento de nitidez
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Normalización de histograma
        for i in range(3):
            enhanced[:,:,i] = cv2.equalizeHist(enhanced[:,:,i])
        
        # Aumento de contraste
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=15)
    
    return enhanced

def get_best_face(faces, boxes, img_shape):
    """Selecciona la mejor cara entre múltiples detecciones"""
    if faces is None or len(faces) == 0:
        return None
    
    if len(faces) == 1:
        return faces[0]
    
    # Criterios para la mejor cara: tamaño y centralidad
    best_score = -1
    best_idx = 0
    center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
    
    for i, box in enumerate(boxes):
        # Calcular tamaño
        width = box[2] - box[0]
        height = box[3] - box[1]
        size = width * height
        
        # Calcular centralidad
        face_center_x = (box[0] + box[2]) // 2
        face_center_y = (box[1] + box[3]) // 2
        distance_from_center = np.sqrt((center_x - face_center_x)**2 + (center_y - face_center_y)**2)
        
        # Score combinado (más grande y más central = mejor)
        score = size / (1 + 0.1 * distance_from_center)
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return faces[best_idx]

def extract_face(img_path, is_ine=False):
    """Extrae rostro con estrategia mejorada"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            return None
        
        # Preprocesamiento adaptado
        img = adaptive_preprocess(img, is_ine)
        
        # Convertir a PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Intento 1: Detección estándar
        boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)
        if boxes is None:
            # Intento 2: Escalar imagen
            larger_img = img_pil.resize((img_pil.width*2, img_pil.height*2))
            boxes, probs, landmarks = mtcnn.detect(larger_img, landmarks=True)
            
            if boxes is None:
                # Intento 3: Más preprocesamiento y rotación
                img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                for angle in [0, -10, 10, -20, 20]:
                    if angle != 0:
                        rotated = img_pil.rotate(angle, expand=True)
                    else:
                        rotated = img_pil
                    
                    boxes, probs, landmarks = mtcnn.detect(rotated, landmarks=True)
                    if boxes is not None:
                        break
        
        if boxes is None:
            return None
            
        # Extraer todas las caras
        faces = mtcnn(img_pil)
        
        # Seleccionar la mejor cara
        best_face = get_best_face(faces, boxes, img.shape)
        
        return best_face
        
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")
        return None

def calculate_similarity(emb1, emb2):
    """Calcula múltiples métricas de similitud y las combina"""
    # Similitud coseno (normalizada entre 0-1)
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    cos_sim = (cos_sim + 1) / 2  # Normalizar de [-1,1] a [0,1]
    
    # Distancia euclidiana (normalizada e invertida)
    eucl_dist = np.linalg.norm(emb1 - emb2)
    eucl_sim = 1 / (1 + eucl_dist)  # Convertir a similitud
    
    # Distancia L1 (normalizada e invertida)
    l1_dist = np.sum(np.abs(emb1 - emb2))
    l1_sim = 1 / (1 + l1_dist)
    
    # Combinación ponderada
    combined_sim = 0.6 * cos_sim + 0.3 * eucl_sim + 0.1 * l1_sim
    
    return {
        'cosine': cos_sim,
        'euclidean': eucl_sim,
        'manhattan': l1_sim,
        'combined': combined_sim
    }

def process_images():
    """Procesa las imágenes y genera dataset"""
    datasets = {
        'same': 1,
        'different': 0
    }
    
    # Estadísticas de procesamiento
    stats = {
        'total': 0,
        'successful': 0,
        'failed': 0
    }
    
    data = []
    
    for folder, label in datasets.items():
        folder_path = os.path.join('datasets', folder)
        
        # Verificar que el directorio existe
        if not os.path.exists(folder_path):
            print(f"El directorio no existe.")
            continue
            
        files = [f for f in os.listdir(folder_path) if '_foto' in f]
        stats['total'] += len(files)
        
        for file in tqdm(files, desc=f'Procesando {folder}'):
            base = file.replace('_foto.jpg', '')
            foto_path = os.path.join(folder_path, file)
            ine_path = os.path.join(folder_path, f'{base}_ine.jpg')
            
            # Verificar que ambos archivos existen
            if not os.path.exists(foto_path) or not os.path.exists(ine_path):
                print(f"Archivos faltantes para {base}")
                stats['failed'] += 1
                continue
            
            # Extraer embeddings
            face_foto = extract_face(foto_path)
            face_ine = extract_face(ine_path, is_ine=True)
            
            if face_foto is None or face_ine is None:
                print(f"No se pudo extraer cara para {base}")
                stats['failed'] += 1
                continue
                
            with torch.no_grad():
                emb_foto = resnet(face_foto.unsqueeze(0).to(device)).cpu().numpy().flatten()
                emb_ine = resnet(face_ine.unsqueeze(0).to(device)).cpu().numpy().flatten()
            
            # Calcular múltiples métricas de similitud
            similarities = calculate_similarity(emb_foto, emb_ine)
            
            # Guardar datos
            entry = {
                'persona': base,
                'label': label,
            }
            
            # Añadir todas las similitudes
            for sim_name, sim_value in similarities.items():
                entry[f'similitud_{sim_name}'] = sim_value
            
            # Añadir embeddings si se desea
            # (Opcional: puede comentar estas líneas si el CSV se vuelve demasiado grande)
            entry.update({f'foto_{i}': val for i, val in enumerate(emb_foto)})
            entry.update({f'ine_{i}': val for i, val in enumerate(emb_ine)})
            
            data.append(entry)
            stats['successful'] += 1
    
    # Imprimir estadísticas
    print(f"\nEstadísticas del procesamiento:")
    print(f"- Total de pares: {stats['total']}")
    print(f"- Procesados exitosamente: {stats['successful']}")
    print(f"- Fallidos: {stats['failed']}")
    
    return pd.DataFrame(data)

def evaluate_model(df):
    """Evalúa diferentes métricas de similitud y genera visualizaciones"""
    metrics = ['cosine', 'euclidean', 'manhattan', 'combined']
    plt.figure(figsize=(15, 10))
    
    # Crear subplots
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # Histogramas
        sns.histplot(df[df.label == 1][f'similitud_{metric}'], 
                   bins=30, alpha=0.5, label='Misma persona', color='green')
        sns.histplot(df[df.label == 0][f'similitud_{metric}'], 
                   bins=30, alpha=0.5, label='Diferente persona', color='red')
        
        plt.xlabel(f'Similitud ({metric})')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.title(f'Distribución de similitud {metric}')
    
    plt.tight_layout()
    plt.savefig('metrics/similitud_distribuciones.png')
    
    # Curvas ROC
    plt.figure(figsize=(10, 8))
    
    for metric in metrics:
        y_true = df['label']
        y_score = df[f'similitud_{metric}']
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = {roc_auc:.3f})')
        
        # Encontrar el mejor umbral (máxima separación entre TPR y FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"Métrica {metric}:")
        print(f"- AUC: {roc_auc:.3f}")
        print(f"- Umbral óptimo: {optimal_threshold:.3f}")
        print(f"- TPR en umbral óptimo: {tpr[optimal_idx]:.3f}")
        print(f"- FPR en umbral óptimo: {fpr[optimal_idx]:.3f}")
        print()
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC para diferentes métricas de similitud')
    plt.legend(loc="lower right")
    plt.savefig('metrics/roc_curves.png')
    
    return {
        'metrics': metrics,
        'best_metric': metrics[np.argmax([auc(roc_curve(df['label'], df[f'similitud_{m}'])[0], 
                                               roc_curve(df['label'], df[f'similitud_{m}'])[1]) 
                                          for m in metrics])]
    }

def start_process():
    # Crear directorios si no existen
    os.makedirs('datasets', exist_ok=True)
    
    # Procesar imágenes
    df = process_images()
    
    # Guardar resultados
    df.to_csv('datasets/embeddings.csv', index=False)
    
    # Evaluar modelo
    if len(df) > 0:
        results = evaluate_model(df)
        
        print(f"\nResultados de la evaluación:")
        print(f"- Mejor métrica: {results['best_metric']}")
        
        # Guardar solo las columnas importantes para un archivo más pequeño
        df_small = df[['persona', 'label'] + [f'similitud_{m}' for m in results['metrics']]]
        df_small.to_csv('datasets/similitudes.csv', index=False)
        
        print("\nArchivos generados:")
        print("- datasets/embeddings.csv (dataset completo con embeddings)")
        print("- datasets/similitudes.csv (dataset reducido con solo similitudes)")
        print("- metrics/similitud_distribuciones.png (gráficos de distribución)")
        print("- metrics/roc_curves.png (curvas ROC)")
    else:
        print("\nNo se pudieron procesar imágenes. Verifica las rutas y archivos.")