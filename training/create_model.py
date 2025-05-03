import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
import os

class FaceVerifier:
    def __init__(self, model_path=None, threshold=0.705):
        # Dispositivo (GPU o CPU)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
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
        
        # Umbral de similitud para verificación directa
        self.threshold = threshold
        
        # Modelo de clasificación (opcional)
        self.classifier = None
        self.feature_names = None
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                # Comprobar si es un diccionario con modelo y nombres de características
                if isinstance(model_data, dict) and 'model' in model_data and 'feature_names' in model_data:
                    self.classifier = model_data['model']
                    self.feature_names = model_data['feature_names']
                else:
                    # Modelo antiguo (solo el clasificador)
                    self.classifier = model_data
                    # Suponer características por defecto
                    self.feature_names = ['cosine', 'euclidean', 'manhattan', 'combined']
                
                print(f"Modelo cargado desde: {model_path}")
                print(f"Características esperadas: {self.feature_names}")
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
    
    def enhance_image(self, img):
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
    
    def adaptive_preprocess(self, img, is_ine=False):
        """Preprocesamiento adaptativo mejorado"""
        # Conversión inicial
        if len(img.shape) == 2:  # Es escala de grises
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Mejora básica
        enhanced = self.enhance_image(img)
        
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
    
    def get_best_face(self, faces, boxes, img_shape):
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
    
    def extract_face(self, img_path, is_ine=False):
        """Extrae rostro con estrategia mejorada"""
        try:
            if isinstance(img_path, str):  # Es una ruta
                img = cv2.imread(img_path)
            else:  # Es una imagen ya cargada
                img = img_path
                
            if img is None:
                print(f"No se pudo procesar la imagen")
                return None
            
            # Preprocesamiento adaptado
            img = self.adaptive_preprocess(img, is_ine)
            
            # Convertir a PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Intento 1: Detección estándar
            boxes, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
            if boxes is None:
                # Intento 2: Escalar imagen
                larger_img = img_pil.resize((img_pil.width*2, img_pil.height*2))
                boxes, probs, landmarks = self.mtcnn.detect(larger_img, landmarks=True)
                
                if boxes is None:
                    # Intento 3: Más preprocesamiento y rotación
                    img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
                    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                    for angle in [0, -10, 10, -20, 20]:
                        if angle != 0:
                            rotated = img_pil.rotate(angle, expand=True)
                        else:
                            rotated = img_pil
                        
                        boxes, probs, landmarks = self.mtcnn.detect(rotated, landmarks=True)
                        if boxes is not None:
                            break
            
            if boxes is None:
                return None
                
            # Extraer todas las caras
            faces = self.mtcnn(img_pil)
            
            # Seleccionar la mejor cara
            best_face = self.get_best_face(faces, boxes, img.shape)
            
            return best_face
            
        except Exception as e:
            print(f"Error procesando la imagen: {e}")
            return None
    
    def get_embedding(self, face_tensor):
        """Obtiene el embedding facial de un tensor"""
        if face_tensor is None:
            return None
            
        with torch.no_grad():
            embedding = self.resnet(face_tensor.unsqueeze(0).to(self.device))
            return embedding.cpu().numpy().flatten()
    
    def calculate_similarity(self, emb1, emb2):
        """Calcula múltiples métricas de similitud"""
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
    
    def verify(self, img1_path, img2_path, is_ine2=True, return_details=False):
        """Verifica si dos imágenes pertenecen a la misma persona"""
        # Extraer caras
        face1 = self.extract_face(img1_path, is_ine=False)
        face2 = self.extract_face(img2_path, is_ine=is_ine2)
        
        if face1 is None or face2 is None:
            print("No se pudieron detectar caras en ambas imágenes")
            return False if not return_details else {
                'match': False, 
                'error': 'No se pudieron detectar caras en ambas imágenes',
                'similarity': None
            }
        
        # Obtener embeddings
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)
        
        # Calcular similitudes
        similarities = self.calculate_similarity(emb1, emb2)
        
        # Verificar usando el modelo entrenado si está disponible
        if self.classifier is not None and self.feature_names is not None:
            # Preparar features para el clasificador
            features = np.array([similarities[feat_name] for feat_name in self.feature_names]).reshape(1, -1)
            
            try:
                # Predecir con el clasificador
                prediction = self.classifier.predict(features)[0]
                probability = self.classifier.predict_proba(features)[0][1]
                
                if return_details:
                    return {
                        'match': bool(prediction),
                        'probability': float(probability),
                        'similarity': similarities
                    }
                else:
                    return bool(prediction)
            except Exception as e:
                print(f"Error al predecir: {e}")
                print(f"Usando umbral simple como fallback")
                # Caer en el método de umbral
        
        # Si no hay clasificador o falló, usar umbral directo en similitud coseno
        is_match = similarities['cosine'] >= self.threshold
        
        if return_details:
            return {
                'match': is_match,
                'probability': float(similarities['cosine']),
                'similarity': similarities
            }
        else:
            return is_match

# Ejemplo de uso
if __name__ == "__main__":
    
    # Verificar: python face_verification_fixed.py verify foto.jpg ine.jpg [modelo.pkl]
    test_path = os.path.join(os.path.dirname(__file__), 'photo_test/')
    img1_path = (f"{test_path}1_foto.jpg")
    img2_path = (f"{test_path}1_ine.jpg")
    model_path = "models/face_verification_model.pkl"
    
    verifier = FaceVerifier(model_path=model_path)
    result = verifier.verify(img1_path, img2_path, return_details=True)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        if result['match']:
            print(f"✅ Las imágenes pertenecen a la misma persona (probabilidad: {result['probability']:.4f})")
        else:
            print(f"❌ Las imágenes no pertenecen a la misma persona (probabilidad: {result['probability']:.4f})")
        
        print("\nDetalles de similitud:")
        for metric, value in result['similarity'].items():
            print(f"- {metric}: {value:.4f}")
      
   