import os
import pandas as pd
#import utils.create_dataset as create_dataset
import utils.create_dataset as create_dataset
import training.train_model as train_model
import training.analyze as analyze_model
from training.use import FaceVerifier

class Main:
    def __init__(self):
        self.name = "Main Class"
        self.options = {
            '1': self.option1,
            '2': self.option2,
            '3': self.option3,
            '4': self.option4,
            '0': self.exit,
        }
        self.running = True
        self.dataset_path = os.path.join(os.path.dirname(__file__), 'datasets/embeddings.csv')
        self.model_path = os.path.join(os.path.dirname(__file__), 'models/face_verification_model.pkl')
        self.test_path = os.path.join(os.path.dirname(__file__), 'photo_test/')
        
    def option1(self):
        # Código para crear dataset
        create_dataset.start_process()
        
    def option2(self):
        # Código para entrenar modelo
        train_model.train_verification_model(self.dataset_path, self.model_path)
        
    def option3(self):
        # Código para analizar modelo
        analyze_model.analyze_model(self.model_path)
        
    def option4(self):
        # Código para usar modelo
        # Inicializar el verificador
        verifier = FaceVerifier(model_path='face_verification_model.pkl')
        
        # Validar si la carpeta existe
        if not os.path.isdir(self.test_path):
            print(f"\nError: La carpeta '{self.test_path}' no existe.")
            return
            
        # Procesar la carpeta
        print(f"\nProcesando imágenes")
        results = self.process_verification_folder(self.test_path, verifier)
        
        # Mostrar resumen
        if results is not None and not results.empty:
            matches = results['match'].sum()
            total = len(results)
            print(f"\n{matches}/{total} coincidencias ({matches/total:.1%})")
            
            # Mostrar tabla resumen si hay pocos resultados
            if total <= 10:
                print("\nResultados:")
                print(results[['foto', 'ine', 'match', 'probability']].to_string(index=False))
                
    def exit(self):
        self.running = False

    def show_menu(self):
        print("\n")
        print("1. Crear dataset")
        print("2. Entrenar modelo")
        print("3. Analizar modelo")
        print("4. Usar modelo")
        print("0. Salir")

    def run(self):
        while self.running:
            self.show_menu()
            choice = input("Selecciona una opción: ")
            action = self.options.get(choice)
            if action:
                action()
            else:
                print("Opción no válida. Intenta de nuevo.")
    
    def process_verification_folder(self, folder_path, verifier):
        """Procesa todos los pares de fotos en una carpeta (adaptado para la clase)"""
        try:
            results = []
            file_pairs = []
            
            # Identificar archivos
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Encontrar pares (foto + INE)
            foto_files = [f for f in files if '_foto.' in f.lower()]
            
            for foto in foto_files:
                prefix = foto.split('_foto.')[0]
                ine_candidates = [f for f in files if f.startswith(f"{prefix}_ine.")]
                
                if ine_candidates:
                    file_pairs.append((foto, ine_candidates[0]))
            
            if not file_pairs:
                print("\nNo se encontraron pares válidos (formato esperado: '1_foto.jpg', '1_ine.jpg')")
                return None
            
            print(f"\nEncontrados {len(file_pairs)} pares de imágenes:")
            for i, (foto, ine) in enumerate(file_pairs, 1):
                print(f" {i}. {foto} ↔ {ine}")
            
            print("\nIniciando verificación...")
            
            for foto, ine in file_pairs:
                foto_path = os.path.join(folder_path, foto)
                ine_path = os.path.join(folder_path, ine)
                
                result = verifier.verify_faces(foto_path, ine_path, is_ine=True, return_details=True)
                
                row = {
                    'foto': foto,
                    'ine': ine,
                    'match': result['match'],
                    'probability': result['probability'],
                    'cosine_sim': result['similarity']['cosine'],
                    'error': result.get('error', '')
                }
                results.append(row)
                
                status = "✅" if row['match'] else "❌"
                print(f" {status} {foto:<15} ↔ {ine:<15} | Prob: {row['probability']:.1%} | Coseno: {row['cosine_sim']:.3f}")
            
            # Crear DataFrame
            df = pd.DataFrame(results)
            
            return df
        
        except Exception as e:
            print(f"\nError durante el procesamiento: {str(e)}")
            return None


if __name__ == "__main__":
    app = Main()
    app.run()
    
