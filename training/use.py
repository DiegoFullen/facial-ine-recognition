import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
import os
import pandas as pd
from datetime import datetime

class FaceVerifier:
    def __init__(self, model_path=None, threshold=0.705):
        # Configuración del dispositivo
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Modelo de detección de caras
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            min_face_size=50,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True
        )
        
        # Modelo de embeddings faciales
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Umbral de similitud
        self.threshold = threshold
        
        # Modelo de clasificación (opcional)
        self.classifier = None
        self.feature_names = None
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Carga el modelo de clasificación"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            if isinstance(model_data, dict) and 'model' in model_data:
                self.classifier = model_data['model']
                self.feature_names = model_data.get('feature_names', ['cosine', 'euclidean', 'manhattan', 'combined'])
            else:
                self.classifier = model_data
                self.feature_names = ['cosine', 'euclidean', 'manhattan', 'combined']
                
            print(f"Modelo cargado")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
    
    def verify_faces(self, img1_path, img2_path, is_ine=False, return_details=False):
        """Verifica si dos imágenes pertenecen a la misma persona"""
        
        # Extraer caras
        face1 = self._extract_face(img1_path, is_ine=False)
        face2 = self._extract_face(img2_path, is_ine=is_ine)
        
        if face1 is None or face2 is None:
            result = {'match': False, 'error': 'No se detectaron caras en ambas imágenes'}
            return result if return_details else False
        
        # Obtener embeddings
        emb1 = self._get_embedding(face1)
        emb2 = self._get_embedding(face2)
        
        # Calcular similitudes
        similarities = self._calculate_similarity(emb1, emb2)
        
        # Determinar coincidencia
        if self.classifier is not None and self.feature_names is not None:
            try:
                features = np.array([similarities[feat] for feat in self.feature_names]).reshape(1, -1)
                prediction = self.classifier.predict(features)[0]
                probability = self.classifier.predict_proba(features)[0][1]
                
                result = {
                    'match': bool(prediction),
                    'probability': float(probability),
                    'similarity': similarities
                }
            except Exception as e:
                print(f"Error al predecir: {e}. Usando umbral simple.")
                result = {
                    'match': similarities['cosine'] >= self.threshold,
                    'probability': float(similarities['cosine']),
                    'similarity': similarities
                }
        else:
            result = {
                'match': similarities['cosine'] >= self.threshold,
                'probability': float(similarities['cosine']),
                'similarity': similarities
            }
        
        return result if return_details else result['match']
    
    def _extract_face(self, img_path, is_ine=False):
        """Extrae el rostro principal de una imagen"""
        try:
            img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
            if img is None:
                return None
                
            # Preprocesamiento básico
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            
            # Detección de caras
            boxes, _, _ = self.mtcnn.detect(img_pil, landmarks=True)
            if boxes is None:
                return None
                
            # Extraer la cara principal
            faces = self.mtcnn(img_pil)
            return faces[0] if faces is not None else None
            
        except Exception as e:
            print(f"Error al extraer cara: {e}")
            return None
    
    def _get_embedding(self, face_tensor):
        """Obtiene el embedding facial de un tensor"""
        with torch.no_grad():
            return self.resnet(face_tensor.unsqueeze(0).to(self.device)).cpu().numpy().flatten()
    
    def _calculate_similarity(self, emb1, emb2):
        """Calcula métricas de similitud entre embeddings"""
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        cos_sim = (cos_sim + 1) / 2  # Normalizar a [0,1]
        
        eucl_dist = np.linalg.norm(emb1 - emb2)
        eucl_sim = 1 / (1 + eucl_dist)
        
        l1_dist = np.sum(np.abs(emb1 - emb2))
        l1_sim = 1 / (1 + l1_dist)
        
        combined_sim = 0.6 * cos_sim + 0.3 * eucl_sim + 0.1 * l1_sim
        
        return {
            'cosine': cos_sim,
            'euclidean': eucl_sim,
            'manhattan': l1_sim,
            'combined': combined_sim
        }