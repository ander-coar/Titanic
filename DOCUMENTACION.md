# Documentación del Proyecto Titanic

## 📊 Descripción General

Este proyecto implementa un análisis completo de datos del famoso dataset del Titanic. Incluye:

- Análisis exploratorio de datos (EDA)
- Visualizaciones estadísticas
- Preprocesamiento de datos
- Modelos de machine learning
- Evaluación y comparación de modelos

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

O si estás en Linux/Ubuntu:

```bash
sudo apt-get install python3-pandas python3-numpy python3-matplotlib python3-sklearn
```

## 📖 Guía de Uso

### Opción 1: Ejecutar el Análisis Completo

La forma más sencilla de ejecutar el análisis completo:

```bash
python3 run_analysis.py
```

Este script ejecutará todo el pipeline de análisis y generará todos los archivos de salida.

### Opción 2: Ejecutar Paso a Paso

#### 1. Descargar/Crear el Dataset

```bash
python3 src/download_data.py
```

Este comando creará el archivo `data/titanic.csv` con los datos del Titanic.

#### 2. Ejecutar el Análisis

```bash
python3 src/analisis_titanic.py
```

Este comando ejecutará el análisis completo y generará:
- Visualizaciones en `visualizations/`
- Reporte de resultados en `ANALISIS_RESULTADOS.txt`

### Opción 3: Usar Jupyter Notebook

Para análisis interactivo:

```bash
jupyter notebook notebooks/analisis_titanic.ipynb
```

## 📁 Estructura del Proyecto

```
-Datos-Titanic-/
│
├── data/                           # Datos del proyecto
│   └── titanic.csv                 # Dataset del Titanic
│
├── src/                            # Código fuente
│   ├── download_data.py           # Script para crear/descargar datos
│   └── analisis_titanic.py        # Script principal de análisis
│
├── notebooks/                      # Jupyter notebooks
│   └── analisis_titanic.ipynb     # Notebook interactivo
│
├── visualizations/                 # Visualizaciones generadas
│   ├── analisis_titanic.png       # Análisis exploratorio
│   ├── correlaciones.png          # Mapa de correlaciones
│   └── evaluacion_modelos.png     # Comparación de modelos
│
├── run_analysis.py                 # Script de ejecución rápida
├── requirements.txt                # Dependencias del proyecto
├── ANALISIS_RESULTADOS.txt        # Reporte de resultados
├── DOCUMENTACION.md               # Este archivo
└── README.md                       # Información general del proyecto
```

## 🔍 Componentes del Análisis

### 1. Carga de Datos

El dataset contiene información sobre los pasajeros del Titanic:

| Campo | Descripción |
|-------|-------------|
| survived | Supervivencia (0 = No, 1 = Sí) |
| pclass | Clase del ticket (1, 2, 3) |
| sex | Sexo del pasajero |
| age | Edad en años |
| sibsp | Número de hermanos/cónyuges a bordo |
| parch | Número de padres/hijos a bordo |
| fare | Tarifa del pasaje |
| embarked | Puerto de embarque (C, Q, S) |

### 2. Análisis Exploratorio (EDA)

El análisis exploratorio incluye:

- **Estadísticas descriptivas**: Media, desviación estándar, cuartiles
- **Valores nulos**: Identificación y manejo de datos faltantes
- **Distribuciones**: Análisis de la distribución de cada variable
- **Tasa de supervivencia**: Análisis global y por grupos

### 3. Visualizaciones

Se generan tres archivos de visualización:

#### a) analisis_titanic.png
Contiene 4 gráficos:
- Distribución de supervivencia
- Tasa de supervivencia por clase
- Distribución de edad
- Supervivencia por sexo

#### b) correlaciones.png
Mapa de calor mostrando correlaciones entre variables numéricas

#### c) evaluacion_modelos.png
Comparación de modelos y matriz de confusión del mejor modelo

### 4. Preprocesamiento

El preprocesamiento incluye:

1. **Tratamiento de valores nulos**:
   - Edad: Rellenado con la mediana
   - Embarked: Rellenado con la moda
   - Deck: Eliminado (muchos valores nulos)

2. **Codificación de variables**:
   - Sex: Conversión a numérico (male=0, female=1)
   - Embarked: One-Hot Encoding (embarked_C, embarked_Q, embarked_S)

3. **Selección de características**:
   - pclass, sex, age, sibsp, parch, fare, embarked_*

4. **División de datos**:
   - 80% entrenamiento
   - 20% prueba

### 5. Modelos de Machine Learning

Se implementan dos modelos:

#### Random Forest Classifier
- Ensemble de árboles de decisión
- n_estimators = 100
- Buen manejo de datos no lineales

#### Logistic Regression
- Modelo lineal clásico
- Interpretable
- Eficiente para clasificación binaria

### 6. Evaluación

Los modelos se evalúan con:

- **Precisión (Accuracy)**: Porcentaje de predicciones correctas
- **Precision**: Proporción de positivos correctos
- **Recall**: Proporción de positivos identificados
- **F1-Score**: Media armónica de precision y recall
- **Matriz de Confusión**: Visualización de errores

## 📊 Interpretación de Resultados

### Hallazgos Principales

1. **Supervivencia por Sexo**:
   - Las mujeres tuvieron una tasa de supervivencia significativamente mayor
   - Esto refleja la política de "mujeres y niños primero"

2. **Supervivencia por Clase**:
   - Primera clase: Mayor tasa de supervivencia
   - Tercera clase: Menor tasa de supervivencia
   - La clase socioeconómica fue un factor determinante

3. **Edad**:
   - Los niños tuvieron mayor probabilidad de supervivencia
   - Los adultos mayores tuvieron menor probabilidad

4. **Familiares a Bordo**:
   - Tener familiares a bordo afectó la supervivencia
   - Tanto sibsp como parch mostraron correlaciones

### Rendimiento de Modelos

Los modelos típicamente logran:
- Precisión: 75-85%
- Mejor rendimiento en predecir no-supervivientes
- Variables más importantes: sex, pclass, age

## 🔧 Personalización

### Modificar Parámetros del Modelo

Edita `src/analisis_titanic.py`:

```python
# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,    # Cambia el número de árboles
    max_depth=10,        # Agrega profundidad máxima
    random_state=42
)

# Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000,       # Cambia iteraciones
    C=1.0,              # Agrega regularización
    random_state=42
)
```

### Agregar Nuevos Modelos

```python
from sklearn.svm import SVC

# En el método train_models()
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(self.X_train, self.y_train)
svm_pred = svm_model.predict(self.X_test)
svm_accuracy = accuracy_score(self.y_test, svm_pred)

self.models['SVM'] = {
    'model': svm_model,
    'predictions': svm_pred,
    'accuracy': svm_accuracy
}
```

### Crear Nuevas Visualizaciones

```python
# En el método visualize_data()
fig, ax = plt.subplots(figsize=(10, 6))

# Tu código de visualización aquí
ax.plot(...)
ax.set_title('Mi Nueva Visualización')

plt.savefig('visualizations/mi_visualizacion.png', dpi=300)
```

## 🐛 Solución de Problemas

### Error: ModuleNotFoundError

**Problema**: `ModuleNotFoundError: No module named 'pandas'`

**Solución**:
```bash
pip install pandas numpy matplotlib scikit-learn
```

O en Ubuntu/Debian:
```bash
sudo apt-get install python3-pandas python3-numpy python3-matplotlib python3-sklearn
```

### Error: FileNotFoundError

**Problema**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/titanic.csv'`

**Solución**:
```bash
python3 src/download_data.py
```

### Error: Permission Denied

**Problema**: `PermissionError: [Errno 13] Permission denied`

**Solución**:
```bash
chmod +x run_analysis.py
# O ejecuta con python directamente
python3 run_analysis.py
```

## 📚 Referencias

- **Dataset**: Kaggle Titanic Dataset
- **Pandas**: https://pandas.pydata.org/
- **Scikit-learn**: https://scikit-learn.org/
- **Matplotlib**: https://matplotlib.org/

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/NuevaCaracteristica`)
3. Commit tus cambios (`git commit -m 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/NuevaCaracteristica`)
5. Abre un Pull Request

## 📝 Notas Adicionales

- Los datos son una muestra del dataset original del Titanic
- Los modelos pueden mejorar con más datos y feature engineering
- El proyecto está diseñado con fines educativos
- Se puede extender con más modelos y técnicas de análisis

## 📧 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Última actualización**: 2025-10-17
**Versión**: 1.0
