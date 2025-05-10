import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class TitanicModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_path = "app/models/titanic_model.joblib"
        self.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    def prepare_data(self):
        # Datos de entrenamiento simplificados del Titanic
        data = {
            'Survived': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            'Pclass': [3, 1, 3, 1, 3, 1, 3, 3, 2, 3, 3, 1, 3, 3, 2, 2, 3, 2, 3, 2],
            'Sex': ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'male'],
            'Age': [22.0, 38.0, 26.0, 35.0, 35.0, np.nan, 54.0, 2.0, 27.0, 14.0, 4.0, 58.0, 20.0, 39.0, 14.0, 55.0, 2.0, 31.0, np.nan, 35.0],
            'SibSp': [1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 3, 0, 0, 0],
            'Parch': [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0],
            'Fare': [7.25, 71.28, 7.92, 53.1, 8.05, 51.86, 26.0, 21.07, 11.5, 30.07, 16.7, 26.55, 8.05, 31.28, 30.0, 16.0, 20.58, 28.5, 7.88, 13.0],
            'Embarked': ['S', 'C', 'S', 'S', 'S', 'S', 'C', 'S', 'C', 'C', 'S', 'C', 'S', 'S', 'C', 'S', 'S', 'S', 'S', 'S']
        }
        df = pd.DataFrame(data)
        
        X = df[self.features]
        y = df['Survived']
        
        return X, y
    
    def train(self):
        X, y = self.prepare_data()
        
        # Definir los preprocesadores para diferentes tipos de columnas
        numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_features = ['Pclass', 'Sex', 'Embarked']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combinar los preprocesadores
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Crear y entrenar el modelo
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        self.model.fit(X, y)
        
        # Guardar el modelo entrenado
        joblib.dump(self.model, self.model_path)
        print("Modelo entrenado y guardado en", self.model_path)
    
    def load(self):
        try:
            self.model = joblib.load(self.model_path)
            print("Modelo cargado desde", self.model_path)
            return True
        except:
            print("No se pudo cargar el modelo. Entrenando uno nuevo...")
            self.train()
            return True
    
    def predict(self, passenger_data):
        """
        Realiza una predicción para un pasajero
        
        Args:
            passenger_data (dict): Datos del pasajero con las características necesarias
            
        Returns:
            int: 1 si el pasajero sobrevive, 0 si no sobrevive
            float: Probabilidad de supervivencia
        """
        if self.model is None:
            self.load()
        
        # Convertir los datos de entrada en un DataFrame
        input_df = pd.DataFrame([passenger_data])
        
        # Asegurarse de que el DataFrame tiene todas las columnas necesarias
        for feature in self.features:
            if feature not in input_df.columns:
                input_df[feature] = None
        
        # Seleccionar solo las características relevantes
        input_df = input_df[self.features]
        
        # Hacer la predicción
        prediction = int(self.model.predict(input_df)[0])
        probability = float(self.model.predict_proba(input_df)[0][1])
        
        return prediction, probability

# Si se ejecuta este archivo directamente, entrenar el modelo
if __name__ == "__main__":
    model = TitanicModel()
    model.train() 