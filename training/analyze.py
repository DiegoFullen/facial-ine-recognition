import pickle

def analyze_model(model_path):
    """Analiza un modelo guardado para entender sus requisitos de características"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            feature_names = model_data.get('feature_names', ['desconocido'])
            print(f"\nModelo: {type(model).__name__}")
            print(f"Nombres de características: {feature_names}")
            print(f"Número de características esperadas: {len(feature_names)}")
            
            # Intentar obtener importancia de características
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                print("\nImportancia de características:")
                for i, (feat, imp) in enumerate(zip(feature_names, importances)):
                    print(f"{i+1}. {feat}: {imp:.4f}")
        else:
            model = model_data
            print(f"Modelo: {type(model).__name__}")
            if hasattr(model, 'n_features_in_'):
                print(f"Número de características esperadas: {model.n_features_in_}")
    except Exception as e:
        print(f"Error al analizar el modelo: {e}")