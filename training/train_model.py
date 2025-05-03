import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

    
def train_verification_model(dataset, output_model_path):
    """Entrena un modelo de verificación facial usando Random Forest"""
    # Cargar datos
    df = pd.read_csv(dataset)
    
    # Verificar columnas necesarias
    similarity_cols = [col for col in df.columns if col.startswith('similitud_')]
    if len(similarity_cols) == 0:
        print("El archivo CSV no contiene columnas de similitud")
        return False
    
    print(f"\nColumnas de similitud encontradas: {similarity_cols}")
    
    # Preparar datos para entrenamiento
    X = df[similarity_cols].values
    y = df['label'].values
    
    # Obtener nombres de características sin el prefijo 'similitud_'
    feature_names = [col.replace('similitud_', '') for col in similarity_cols]
    print(f"Nombres de características: {feature_names}")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    print(f"Precisión del modelo: {accuracy:.4f}")
    print("\nInforme de clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Guardar modelo junto con los nombres de características
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'num_features': len(feature_names)
    }
    
    with open(output_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Modelo guardado en: {output_model_path}")
    print(f"Número de características: {len(feature_names)}")
    return True