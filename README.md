# 🚢 Análisis de Datos del Titanic

Proyecto de análisis de datos del famoso dataset del Titanic de Kaggle. Este proyecto implementa un análisis exploratorio completo, preprocesamiento de datos, y modelos de machine learning para predecir la supervivencia de los pasajeros.

## 📋 Descripción

El hundimiento del RMS Titanic es uno de los naufragios más infames de la historia. Este proyecto analiza datos sobre los pasajeros del Titanic para entender qué factores influyeron en su supervivencia.

## 🎯 Objetivos

- Realizar un análisis exploratorio de datos (EDA) completo
- Identificar patrones y correlaciones en los datos
- Preprocesar y limpiar los datos
- Construir modelos predictivos de supervivencia
- Evaluar y comparar diferentes algoritmos de machine learning

## 📊 Dataset

El dataset contiene información sobre los pasajeros del Titanic incluyendo:

- **survived**: Supervivencia (0 = No, 1 = Sí)
- **pclass**: Clase del ticket (1 = 1ra, 2 = 2da, 3 = 3ra)
- **sex**: Sexo del pasajero
- **age**: Edad en años
- **sibsp**: Número de hermanos/cónyuges a bordo
- **parch**: Número de padres/hijos a bordo
- **fare**: Tarifa del pasaje
- **embarked**: Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **pandas**: Manipulación y análisis de datos
- **numpy**: Operaciones numéricas
- **matplotlib**: Visualización de datos
- **seaborn**: Visualización estadística
- **scikit-learn**: Machine Learning

## 📦 Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/ander-coar/-Datos-Titanic-.git
cd -Datos-Titanic-
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### Descargar los datos

```bash
python src/download_data.py
```

### Ejecutar el análisis completo

```bash
python src/analisis_titanic.py
```

Este script ejecutará:
1. Carga de datos
2. Análisis exploratorio
3. Visualizaciones
4. Preprocesamiento
5. Entrenamiento de modelos
6. Evaluación de resultados

## 📈 Resultados

El análisis genera los siguientes archivos:

- **visualizations/analisis_titanic.png**: Gráficos exploratorios del dataset
- **visualizations/correlaciones.png**: Mapa de calor de correlaciones
- **visualizations/evaluacion_modelos.png**: Comparación de modelos
- **ANALISIS_RESULTADOS.txt**: Reporte completo del análisis

## 🔍 Hallazgos Principales

- Las mujeres tuvieron una tasa de supervivencia significativamente mayor que los hombres
- Los pasajeros de primera clase tuvieron mayor probabilidad de sobrevivir
- La edad y el número de familiares a bordo también influyeron en la supervivencia
- Los modelos de machine learning logran una precisión superior al 80%

## 📁 Estructura del Proyecto

```
-Datos-Titanic-/
├── data/                      # Datos del Titanic
│   └── titanic.csv
├── src/                       # Código fuente
│   ├── download_data.py      # Script para descargar datos
│   └── analisis_titanic.py   # Análisis principal
├── notebooks/                 # Jupyter notebooks (opcional)
├── visualizations/            # Visualizaciones generadas
├── requirements.txt           # Dependencias del proyecto
├── .gitignore                # Archivos ignorados por Git
└── README.md                 # Este archivo
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

## 👤 Autor

**Ander Coar**

## 🙏 Agradecimientos

- Dataset proporcionado por Kaggle
- Comunidad de Data Science por recursos y tutoriales