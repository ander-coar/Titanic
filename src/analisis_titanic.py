"""
Análisis de Datos del Titanic
Este script realiza un análisis completo de los datos del Titanic de Kaggle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
    sns.set_style("whitegrid")
except ImportError:
    SEABORN_AVAILABLE = False
    print("Note: Seaborn not available, using matplotlib only")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.rcParams['figure.figsize'] = (12, 8)

class TitanicAnalysis:
    """Clase para análisis completo del dataset Titanic"""
    
    def __init__(self, data_path='data/titanic.csv'):
        """
        Inicializar el análisis
        
        Args:
            data_path: Ruta al archivo CSV del Titanic
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        
    def load_data(self):
        """Cargar los datos del Titanic"""
        print("="*60)
        print("CARGANDO DATOS DEL TITANIC")
        print("="*60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDimensiones del dataset: {self.df.shape}")
        print(f"Número de registros: {self.df.shape[0]}")
        print(f"Número de características: {self.df.shape[1]}")
        
        print("\nPrimeras filas del dataset:")
        print(self.df.head())
        
        return self.df
    
    def exploratory_analysis(self):
        """Análisis exploratorio de datos"""
        print("\n" + "="*60)
        print("ANÁLISIS EXPLORATORIO DE DATOS")
        print("="*60)
        
        # Información general
        print("\nInformación general del dataset:")
        print(self.df.info())
        
        # Estadísticas descriptivas
        print("\nEstadísticas descriptivas:")
        print(self.df.describe())
        
        # Valores nulos
        print("\nValores nulos por columna:")
        null_counts = self.df.isnull().sum()
        print(null_counts[null_counts > 0])
        
        # Distribución de supervivencia
        print("\nDistribución de supervivencia:")
        survival_counts = self.df['survived'].value_counts()
        print(survival_counts)
        print(f"Tasa de supervivencia: {self.df['survived'].mean():.2%}")
        
        return self.df.describe()
    
    def visualize_data(self):
        """Crear visualizaciones del dataset"""
        print("\n" + "="*60)
        print("GENERANDO VISUALIZACIONES")
        print("="*60)
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribución de supervivencia
        survival_counts = self.df['survived'].value_counts()
        axes[0, 0].bar(['No Sobrevivió', 'Sobrevivió'], survival_counts.values, 
                       color=['#d62728', '#2ca02c'])
        axes[0, 0].set_title('Distribución de Supervivencia', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Número de Pasajeros')
        for i, v in enumerate(survival_counts.values):
            axes[0, 0].text(i, v + 10, str(v), ha='center', va='bottom')
        
        # 2. Supervivencia por clase
        survival_by_class = pd.crosstab(self.df['pclass'], self.df['survived'], normalize='index')
        survival_by_class.plot(kind='bar', ax=axes[0, 1], color=['#d62728', '#2ca02c'])
        axes[0, 1].set_title('Tasa de Supervivencia por Clase', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Clase')
        axes[0, 1].set_ylabel('Proporción')
        axes[0, 1].set_xticklabels(['1ra Clase', '2da Clase', '3ra Clase'], rotation=0)
        axes[0, 1].legend(['No Sobrevivió', 'Sobrevivió'])
        
        # 3. Distribución de edad
        axes[1, 0].hist(self.df['age'].dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Distribución de Edad', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Edad')
        axes[1, 0].set_ylabel('Frecuencia')
        axes[1, 0].axvline(self.df['age'].mean(), color='red', linestyle='--', 
                          label=f'Media: {self.df["age"].mean():.1f}')
        axes[1, 0].legend()
        
        # 4. Supervivencia por sexo
        survival_by_sex = pd.crosstab(self.df['sex'], self.df['survived'])
        survival_by_sex.plot(kind='bar', ax=axes[1, 1], color=['#d62728', '#2ca02c'])
        axes[1, 1].set_title('Supervivencia por Sexo', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Sexo')
        axes[1, 1].set_ylabel('Número de Pasajeros')
        axes[1, 1].set_xticklabels(['Femenino', 'Masculino'], rotation=0)
        axes[1, 1].legend(['No Sobrevivió', 'Sobrevivió'])
        
        plt.tight_layout()
        plt.savefig('visualizations/analisis_titanic.png', dpi=300, bbox_inches='tight')
        print("✓ Visualización guardada en: visualizations/analisis_titanic.png")
        
        # Crear mapa de calor de correlaciones
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Seleccionar solo columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        if SEABORN_AVAILABLE:
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, cbar_kws={'label': 'Correlación'})
        else:
            im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.columns)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.columns)
            plt.colorbar(im, ax=ax, label='Correlación')
            
        ax.set_title('Mapa de Calor de Correlaciones', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/correlaciones.png', dpi=300, bbox_inches='tight')
        print("✓ Mapa de correlaciones guardado en: visualizations/correlaciones.png")
        
        plt.close('all')
    
    def preprocess_data(self):
        """Preprocesar datos para el modelo"""
        print("\n" + "="*60)
        print("PREPROCESAMIENTO DE DATOS")
        print("="*60)
        
        # Crear una copia para no modificar el original
        df_processed = self.df.copy()
        
        # Rellenar valores nulos en edad con la mediana
        df_processed['age'].fillna(df_processed['age'].median(), inplace=True)
        print("✓ Valores nulos en 'age' rellenados con la mediana")
        
        # Rellenar valores nulos en embarked con la moda
        if 'embarked' in df_processed.columns:
            df_processed['embarked'].fillna(df_processed['embarked'].mode()[0], inplace=True)
            print("✓ Valores nulos en 'embarked' rellenados con la moda")
        
        # Eliminar columna deck si existe (muchos valores nulos)
        if 'deck' in df_processed.columns:
            df_processed.drop('deck', axis=1, inplace=True)
            print("✓ Columna 'deck' eliminada (muchos valores nulos)")
        
        # Convertir variables categóricas a numéricas
        if 'sex' in df_processed.columns:
            df_processed['sex'] = df_processed['sex'].map({'male': 0, 'female': 1})
            print("✓ Variable 'sex' convertida a numérica")
        
        if 'embarked' in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=['embarked'], prefix='embarked')
            print("✓ Variable 'embarked' convertida con One-Hot Encoding")
        
        # Eliminar columnas no necesarias para el modelo
        columns_to_drop = ['who', 'adult_male', 'alive', 'embark_town', 'class']
        columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if columns_to_drop:
            df_processed.drop(columns_to_drop, axis=1, inplace=True)
            print(f"✓ Columnas eliminadas: {columns_to_drop}")
        
        # Seleccionar características para el modelo
        feature_columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
        feature_columns = [col for col in feature_columns if col in df_processed.columns]
        
        # Agregar columnas de embarked si existen
        embarked_cols = [col for col in df_processed.columns if col.startswith('embarked_')]
        feature_columns.extend(embarked_cols)
        
        X = df_processed[feature_columns]
        y = df_processed['survived']
        
        # Eliminar filas con valores nulos restantes
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]
        
        print(f"\nCaracterísticas seleccionadas: {feature_columns}")
        print(f"Shape final de X: {X.shape}")
        print(f"Shape final de y: {y.shape}")
        
        # Dividir en conjunto de entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nConjunto de entrenamiento: {self.X_train.shape[0]} muestras")
        print(f"Conjunto de prueba: {self.X_test.shape[0]} muestras")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Entrenar modelos de clasificación"""
        print("\n" + "="*60)
        print("ENTRENAMIENTO DE MODELOS")
        print("="*60)
        
        # Modelo 1: Random Forest
        print("\n1. Random Forest Classifier")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        rf_pred = rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)
        
        self.models['Random Forest'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'accuracy': rf_accuracy
        }
        
        print(f"   Precisión: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
        
        # Modelo 2: Logistic Regression
        print("\n2. Logistic Regression")
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(self.X_train, self.y_train)
        lr_pred = lr_model.predict(self.X_test)
        lr_accuracy = accuracy_score(self.y_test, lr_pred)
        
        self.models['Logistic Regression'] = {
            'model': lr_model,
            'predictions': lr_pred,
            'accuracy': lr_accuracy
        }
        
        print(f"   Precisión: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
        
        return self.models
    
    def evaluate_models(self):
        """Evaluar y comparar modelos"""
        print("\n" + "="*60)
        print("EVALUACIÓN DE MODELOS")
        print("="*60)
        
        for model_name, model_data in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            print(f"Precisión: {model_data['accuracy']:.4f}")
            print("\nReporte de Clasificación:")
            print(classification_report(self.y_test, model_data['predictions'], 
                                       target_names=['No Sobrevivió', 'Sobrevivió']))
        
        # Visualizar comparación de modelos
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Comparación de precisión
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        
        axes[0].bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e'])
        axes[0].set_title('Comparación de Precisión de Modelos', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Precisión')
        axes[0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
        
        # Matriz de confusión del mejor modelo
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['accuracy'])
        best_predictions = self.models[best_model_name]['predictions']
        
        cm = confusion_matrix(self.y_test, best_predictions)
        
        if SEABORN_AVAILABLE:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                       xticklabels=['No Sobrevivió', 'Sobrevivió'],
                       yticklabels=['No Sobrevivió', 'Sobrevivió'])
        else:
            im = axes[1].imshow(cm, cmap='Blues')
            axes[1].set_xticks([0, 1])
            axes[1].set_yticks([0, 1])
            axes[1].set_xticklabels(['No Sobrevivió', 'Sobrevivió'])
            axes[1].set_yticklabels(['No Sobrevivió', 'Sobrevivió'])
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
            
        axes[1].set_title(f'Matriz de Confusión - {best_model_name}', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Valor Real')
        axes[1].set_xlabel('Predicción')
        
        plt.tight_layout()
        plt.savefig('visualizations/evaluacion_modelos.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualización de evaluación guardada en: visualizations/evaluacion_modelos.png")
        
        plt.close('all')
        
        # Mostrar el mejor modelo
        print(f"\n{'='*60}")
        print(f"MEJOR MODELO: {best_model_name}")
        print(f"Precisión: {self.models[best_model_name]['accuracy']:.4f}")
        print(f"{'='*60}")
    
    def generate_report(self):
        """Generar reporte final del análisis"""
        print("\n" + "="*60)
        print("GENERANDO REPORTE FINAL")
        print("="*60)
        
        report = []
        report.append("# REPORTE DE ANÁLISIS DEL TITANIC\n")
        report.append("="*60 + "\n")
        
        # Resumen del dataset
        report.append("\n## 1. RESUMEN DEL DATASET\n")
        report.append(f"- Total de registros: {self.df.shape[0]}\n")
        report.append(f"- Número de características: {self.df.shape[1]}\n")
        report.append(f"- Tasa de supervivencia general: {self.df['survived'].mean():.2%}\n")
        
        # Hallazgos principales
        report.append("\n## 2. HALLAZGOS PRINCIPALES\n")
        
        # Supervivencia por sexo
        survival_by_sex = self.df.groupby('sex')['survived'].mean()
        report.append("\n### Supervivencia por Sexo:\n")
        for sex, rate in survival_by_sex.items():
            report.append(f"- {sex.capitalize()}: {rate:.2%}\n")
        
        # Supervivencia por clase
        survival_by_class = self.df.groupby('pclass')['survived'].mean()
        report.append("\n### Supervivencia por Clase:\n")
        for pclass, rate in survival_by_class.items():
            report.append(f"- Clase {pclass}: {rate:.2%}\n")
        
        # Resultados de los modelos
        report.append("\n## 3. RESULTADOS DE MODELOS\n")
        for model_name, model_data in self.models.items():
            report.append(f"\n### {model_name}:\n")
            report.append(f"- Precisión: {model_data['accuracy']:.4f} ({model_data['accuracy']*100:.2f}%)\n")
        
        # Mejor modelo
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['accuracy'])
        report.append(f"\n### Mejor Modelo: {best_model_name}\n")
        report.append(f"- Precisión: {self.models[best_model_name]['accuracy']:.4f}\n")
        
        # Conclusiones
        report.append("\n## 4. CONCLUSIONES\n")
        report.append("- Las mujeres tuvieron una tasa de supervivencia significativamente mayor que los hombres\n")
        report.append("- La clase del pasajero fue un factor determinante en la supervivencia\n")
        report.append("- Los pasajeros de primera clase tuvieron mayor probabilidad de sobrevivir\n")
        report.append("- La edad y el número de familiares a bordo también influyeron en la supervivencia\n")
        
        # Guardar reporte
        with open('ANALISIS_RESULTADOS.txt', 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print("✓ Reporte guardado en: ANALISIS_RESULTADOS.txt")
        
        # Mostrar reporte en consola
        print("\n" + "".join(report))
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("\n" + "="*60)
        print("INICIO DEL ANÁLISIS COMPLETO DEL TITANIC")
        print("="*60)
        
        # Paso 1: Cargar datos
        self.load_data()
        
        # Paso 2: Análisis exploratorio
        self.exploratory_analysis()
        
        # Paso 3: Visualizaciones
        self.visualize_data()
        
        # Paso 4: Preprocesamiento
        self.preprocess_data()
        
        # Paso 5: Entrenar modelos
        self.train_models()
        
        # Paso 6: Evaluar modelos
        self.evaluate_models()
        
        # Paso 7: Generar reporte
        self.generate_report()
        
        print("\n" + "="*60)
        print("ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("\nArchivos generados:")
        print("  - visualizations/analisis_titanic.png")
        print("  - visualizations/correlaciones.png")
        print("  - visualizations/evaluacion_modelos.png")
        print("  - ANALISIS_RESULTADOS.txt")


def main():
    """Función principal"""
    # Crear instancia del análisis
    analysis = TitanicAnalysis()
    
    # Ejecutar análisis completo
    analysis.run_complete_analysis()


if __name__ == "__main__":
    main()
