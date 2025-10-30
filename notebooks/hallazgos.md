# EDA - Análisis Exploratorio de Datos - Hallazgos Clave

El EDA no solo es vizualizar datos

## FASE 1: Análisis Inicial y Hallazgos Clave
- Comprobar unidades de datos , consistencia semantica
- Analizar los tipos de variables (numericas, categóricas, ordinales, temporales)
- Verificar la estructura: filas y columnas, granularidad
- Revisar el tipo de dato (int, float, object, datetime)}

### Comandos mas usados en esta fase
- import pandas as pd
- df = pd.read_csv('ruta/del/archivo.csv')
- df.shape
- df.info()
- df.describe()
- df.head()
- df.columns
- df.dtypes
- df.nunique()

- Cantidad de datos: 891
- Valores faltantes: 0
- Valores únicos (Cardinalidad): 891 # df.nunique() 
- Tipo de variable: Cuantitativa discreta (Identificador unico) 
- Tipo de dato: Entero (int64)
- Descripción: ID único para cada pasajero
- Rango: 1 a 891
- Distribución: Uniforme (incremental de 1 en 1)
- Tipo de dato vacio: Ninguno
- Clasificación de ausencia (MCAR - MAR - MNAR): Ninguno
- Outliers: Ninguno
- Dependencias: Ninguna (No afecta ni es afectada por otras variables)
- Tipo de relación con el target: Ninguna
- Tratamiento previsto: Eliminar antes del modelado
- Relevancia: Nula (No aporta información predictiva)
- Observaciones: Sin errores, datos limpios




### Bitacora de Variables (data dictionary)
Numero de filas: 891
Numero de columnas: 12
- PassengerId:
  - Cantidad de datos: 891
  - Valores faltantes: 0
  - Valores únicos (Cardinalidad): 891 # df.nunique() 
  - Tipo de variable: Cuantitativa discreta (Identificador unico) 
  - Tipo de dato: Entero (int64)
  - Descripción: ID único para cada pasajero
  - Rango: 1 a 891
  - Distribución: Uniforme (incremental de 1 en 1)
  - Tipo de dato vacio: Ninguno
  - Clasificación de ausencia (MCAR - MAR - MNAR): Ninguno
  - Outliers: Ninguno
  - Dependencias: Ninguna (No afecta ni es afectada por otras variables)
  - Tipo de relación con el target: Ninguna
  - Tratamiento previsto: Eliminar antes del modelado
  - Relevancia: Nula (No aporta información predictiva)
  - Observaciones: Sin errores, datos limpios
- Survived:
    - Valores faltantes: 0
    - Valores únicos: 2 (0 - Murio, 1 - Sobrevivio)
    - Tipo de variable: Cuantitativa discreta (binaria)
    - Tipo de dato: Entero (int64) 
    - Descripción: Variable objetivo, indica quien sobrevivió y quien no
    - Relevancia: Alta (Variable objetivo)
- Pclass:
    - Cantidad de datos: 891
    - Valores faltantes: 0
    - Valores únicos: 3 (1 - Primera Clase, 2 - Segunda Clase, 3 - Tercera Clase)
    - Tipo de variable: Cuantitativa discreta
    - Tipo de dato: Entero (int64) 
    - Descripción: Indica la clase del boleto del pasajero
    - Relevancia:
- Name:
    - Cantidad de datos: 891
    - Valores faltantes: 0
    - Valores únicos: 891
    - Tipo de variable: Cualitativa nominal
    - Tipo de dato: Objeto (object) / String
    - Descripción: Indica el nombre completo del pasajero
      - Estructura 1: típica: Apellido, Título. Nombre(s).
      - Estructura 2: Apellido, Título(Mrs). Nombre-Esposo (Nombre Esposa).
        - Apellido: Apellido familiar (a menudo compartido entre familiares o cónyuges).
        - Título: Indica el estatus social o profesional (Mr, Mrs, Miss, Master, Dr, Rev, etc.)
        - Nombre(s): Nombre(s) de pila del pasajero.
        - Nombre en parentésis: A veces incluye el nombre del esposo o un alias.
    - Relevancia: 
- Sex:
  - Cantidad de datos: 891
  - Valores faltantes: 0
  - Valores únicos: 2 (male, female)
  - Tipo de variable: Cualitativa nominal
  - Tipo de dato: Objeto (object) / String
  - Descripción: Indica el sexo del pasajero
  - Relevancia:
- Age:
    - Cantidad de datos: 714
    - Valores faltantes: 177
    - Valores únicos (Cardinalidad): 88
    - Tipo de variable: Cuantitativa continua 
    - Tipo de dato: Flotante (float64)
    - Descripción: Edad del pasajero en años y meses (puede ser fraccionaria)
    - Rango: 0.42 - 80.0
    - Distribución: 
    - Tipo de dato vacio: null
    - Clasificación de ausencia (MCAR - MAR - MNAR): MAR
    - Outliers: Existen outliers en edades avanzadas (>64.8 años)
    - Dependencias: Ninguna (No afecta ni es afectada por otras variables)
    - Tipo de relación con el target: Ninguna
    - Tratamiento previsto: Eliminar antes del modelado
    - Relevancia: Nula (No aporta información predictiva)
    - Observaciones: Sin errores, datos limpios
- SibSp:
    - Cantidad de datos: 891
    - Valores faltantes: 0
    - Valores únicos: 7
    - Tipo de variable: Cuantitativa discreta
    - Tipo de dato: Entero (int64)
    - Descripción: Número de hermanos/cónyuges a bordo (no tiene en cuenta los amantes o prometidos)
    - Relevancia:
- Parch:
    - Cantidad de datos: 891
    - Valores faltantes: 0
    - Valores únicos: 7
    - Tipo de variable: Cuantitativa discreta
    - Tipo de dato: Entero (int64)
    - Descripción: Número de padres/hijos a bordo (no incluye niñeras)
    - Relevancia:
- Ticket:
    - Cantidad de datos: 891
    - Valores faltantes: 0
    - Valores únicos: 681
    - Tipo de variable: Cualitativa nominal
    - Tipo de dato: Objeto (object) / String
    - Descripción: Número de ticket del pasajero
    - Relevancia:
- Fare:
    - Cantidad de datos: 891
    - Valores faltantes: 0
    - Valores únicos: 248
    - Tipo de variable: Cuantitativa continua
    - Tipo de dato: Flotante (float64)
    - Descripción: Tarifa pagada por el pasajero
    - Relevancia:
- Cabin:
    - Cantidad de datos: 204
    - Valores faltantes: 687
    - Valores únicos: 147
    - Tipo de variable: Cualitativa nominal
    - Tipo de dato: Objeto (object) / String
    - Descripción: Número de cabina del pasajero
    - Relevancia:
- Embarked:
    - Cantidad de datos: 889
    - Valores faltantes: 2
    - Valores únicos: 3 (C, Q, S) / Zona o lugar de embarque
    - Tipo de variable: Cualitativa nominal
    - Tipo de dato: Objeto (object) / String
    - Descripción: Puerto de embarque del pasajero (C = Cherburgo, Q = Queenstown, S = Southampton)
    - Relevancia:

## Fase2: Evaluar la calida de los datos (Nombre en ingles Data Quality Assessment - DQA)

- Diagnosticar problemas de calidad de datos:
  - Valores faltantes
  - Valores duplicados
  - Valores inconsistentes o erróneos
  - Outliers o valores atípicos
  - Formato incorrecto de datos

- Age:
  - Valores faltantes: 177
  - Valores inconsistentes: Ninguno detectado
  - Valores Erróneos: Ninguno detectado
  - Outliers: 
  - Valores atipicos:
  - Formato incorrecto:

Cómo identificar Outliers:
Usamos Metodo IQR (Interquartile Range . Rango intercuartil)



## 📊 HALLAZGOS CLAVE - EDA

### Analisis inicial 
Columnas irrelevantes para el análisis inicial:
- La columna Id pasajero no genera valor para el objetivo del proyecto
- la Columna nombre no aporta valor predictivo
- Sexo es relevante para predecir supervivencia
- 

Columnas relevantes para el análisis inicial:
- Pclass
- 

### 1. Variable Objetivo (Survived)
- **38% sobrevivió** vs 62% murió
- Dataset ligeramente desbalanceado
- Baseline (modelo dummy): 62%

### 2. Sex vs Survived ⭐ MUY IMPORTANTE
- **Mujeres: 74%** sobrevivieron (233 de 314)
- **Hombres: 19%** sobrevivieron (109 de 577)
- **Diferencia: 55 puntos porcentuales**
- **Conclusión:** Política "mujeres primero" confirmada
- **Predictividad:** ALTA (la variable más importante)

### 3. Pclass vs Survived ⭐ IMPORTANTE
- **1ra clase: 63%** sobrevivieron (136 de 216)
- **2da clase: 47%** sobrevivieron (87 de 184)
- **3ra clase: 24%** sobrevivieron (119 de 491)
- **Conclusión:** A mayor clase social, mayor supervivencia
- **Predictividad:** ALTA (segunda más importante)

### 4. Decisiones Preliminares
**Eliminar:**
- PassengerId (solo ID): Se debe eliminar antes ya que es una variable identificadora única que no aporta información predictiva.
- Cabin (77% missing): Demasiados valores faltantes para imputar confiablemente.

**Mantener y transformar:**
- Sex ✅
- Pclass ✅
- Age (imputar faltantes)

---
**Próximo:** Analizar Age, Fare, SibSp, Parch, Embarked
```