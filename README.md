# 🚢 PROYECTO TITANIC - APRENDIZAJE DE DATA SCIENCE

## El desafío
El hundimiento del Titanic es uno de los naufragios más infames de la historia.

El 15 de abril de 1912, durante su viaje inaugural, el ampliamente considerado “insumergible” RMS Titanic se hundió tras chocar con un iceberg. Desafortunadamente, no había suficientes botes salvavidas para todos a bordo, lo que provocó la muerte de 1.502 de los 2.224 pasajeros y tripulantes.

Si bien hubo cierto elemento de suerte involucrado en la supervivencia, parece que algunos grupos de personas tenían más probabilidades de sobrevivir que otros.

## Diccionario de datos
| **Variable**     | **Definición**                        | **Clave / Descripción**                                                   |
| ---------------- | ------------------------------------- | ------------------------------------------------------------------------- |
| **supervivencia** | Supervivencia del pasajero            | 0 = No, 1 = Sí                                                            |
| **clasep**       | Clase de billete                      | 1 = 1ª clase, 2 = 2ª clase, 3 = 3ª clase                                  |
| **sexo**         | Sexo del pasajero                     | —                                                                         |
| **edad**         | Edad en años                          | Fraccionaria si es menor que 1; si es estimada, se representa como `xx.5` |
| **sibsp**        | Número de hermanos / cónyuges a bordo | —                                                                         |
| **parch**        | Número de padres / hijos a bordo      | —                                                                         |
| **billete**      | Número de billete del pasajero        | —                                                                         |
| **tarifa**       | Tarifa de pasajero                    | —                                                                         |
| **cabina**       | Número de cabina                      | —                                                                         |
| **embarcado**    | Puerto de embarque                    | C = Cherburgo, Q = Queenstown, S = Southampton                            |


### **Notas variables**
**clasep**: Un indicador del estatus socioeconómico (NSE)
- 1o = Superior
- 2do = Medio
- 3o = Inferior

**edad**: La edad es fraccionaria si es menor que 1. Si se estima la edad, ¿tiene la forma de xx.5

**sibsp**: El conjunto de datos define las relaciones familiares de esta manera...
Hermano = hermano, hermana, hermanastro, hermanastra
Cónyuge = marido, mujer (las amantes y los prometidos fueron ignorados)

**parch**: El conjunto de datos define las relaciones familiares de esta manera...
- Padre = madre, padre
- Niño = hija, hijo, hijastra, hijastro 
- Algunos niños viajaban sólo con una niñera, por lo tanto parch=0 para ellos.


## 📋 ¿Qué es este proyecto?

Mi primer proyecto práctico de **Data Science y Machine Learning** usando el dataset de Kaggle: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic).

## **Objetivo:** 
En este desafío, le pedimos que construya un modelo predictivo que responda a la pregunta: “¿qué tipo de personas tenían más probabilidades de sobrevivir?” utilizando datos de pasajeros (es decir, nombre, edad, sexo, clase socioeconómica, etc.).

**Nivel:** Principiante → Básico  
**Tiempo estimado:** 15-25 horas  
**Dataset:** 891 pasajeros (entrenamiento) + 418 pasajeros (prueba)

---

## 🎯 ¿Qué aprenderé REALMENTE con este proyecto?

### ✅ **LO QUE DOMINARÉ:**

**1. Análisis Exploratorio (EDA)**
- Cargar datos con pandas
- Usar `.head()`, `.info()`, `.describe()`
- Identificar tipos de variables (numéricas, categóricas)
- Detectar valores faltantes
- Crear visualizaciones básicas (histogramas, boxplots, barras)
- Encontrar patrones relacionados con supervivencia

**2. Limpieza de Datos**
- Detectar y contar valores faltantes
- Decidir estrategias de imputación (media, mediana, moda)
- Rellenar valores faltantes sin crear data leakage
- Eliminar columnas irrelevantes

**3. Feature Engineering Básico**
- Extraer información útil (título del nombre)
- Combinar variables existentes (tamaño de familia)
- Crear variables binarias (viaja solo/acompañado)
- Agrupar variables continuas en categorías (bins de edad)

**4. Encoding de Variables Categóricas**
- One-Hot Encoding para nominales (género, puerto)
- Ordinal Encoding para variables con orden (clase social)
- Entender cuándo usar cada método

**5. Modelado Básico**
- Separar datos en train/test correctamente
- Entrenar 3-4 modelos: Logistic Regression, Decision Tree, Random Forest
- Evaluar con accuracy y confusion matrix
- Hacer predicciones en test set

**6. Submission a Kaggle**
- Crear archivo de predicciones en formato correcto
- Subir a Kaggle y obtener score público
- Entender el proceso completo de una competencia

### 🟡 **LO QUE VERÉ PERO NO DOMINARÉ:**

- Cross-validation (k-fold)
- Grid Search básico para optimización
- Feature importance
- Escalado de variables (StandardScaler)
- Más métricas: precision, recall, F1-score

*Estas técnicas las veré y usaré, pero necesito más proyectos para dominarlas*

## 🎓 ¿Qué nivel tendré después del proyecto?

**ANTES:** Principiante absoluto en ML  
**DESPUÉS:** Básico - Junior

### Podré:
✅ Cargar y explorar datasets con pandas  
✅ Limpiar datos básicos  
✅ Entrenar modelos simples de clasificación  
✅ Evaluar modelos correctamente  
✅ Completar el flujo completo de un proyecto de ML  
✅ Entender la lógica de Data Science  

**Tendré bases sólidas para seguir aprendiendo.**


## 📂 Estructura del Proyecto
```
Ejercicio Titanic/
│
├── data/
│   ├── raw/                    # Datos originales
│   │   ├── train.csv          # 891 pasajeros
│   │   └── test.csv           # 418 pasajeros
│   └── processed/              # Datos procesados
│
├── notebooks/                  # Análisis paso a paso
│   ├── 01_eda.ipynb           # Exploración
│   ├── 02_preprocessing.ipynb  # Limpieza
│   ├── 03_modeling.ipynb       # Modelos
│   └── 04_submission.ipynb     # Predicción final
│
├── src/                        # Funciones reutilizables
├── models/                     # Modelos guardados
├── requirements.txt
└── README.md
```

---

## 📊 Progreso

- [x] Setup del proyecto
- [x] Carga de datos inicial
- [ ] EDA completo
- [ ] Limpieza de datos
- [ ] Feature engineering
- [ ] Entrenamiento de modelos
- [ ] Submission a Kaggle
- [ ] Score objetivo: >75% accuracy (realista para primer intento)

---
## 💭 Reflexiones

### ¿Por qué este proyecto?

- Dataset pequeño y manejable para aprender
- Problema clásico con mucha documentación
- Enseña el flujo completo de ML
- Ideal para construir fundamentos sólidos

### ¿Qué sigue después?

1. **House Prices** (Kaggle) - Feature engineering avanzado
2. **Customer Churn** - Datos desbalanceados
3. **Proyecto con texto o imágenes** - Nuevas técnicas

---

## 📝 Nota Importante

**Este es mi PRIMER proyecto de ML.** 

No pretende ser perfecto ni innovador. El objetivo es:
- ✅ Aprender haciendo
- ✅ Entender conceptos fundamentales
- ✅ Cometer errores y aprender de ellos
- ✅ Documentar mi proceso de aprendizaje
- ✅ Tener un punto de partida sólido

*La excelencia viene con la práctica. Esto es solo el inicio.*