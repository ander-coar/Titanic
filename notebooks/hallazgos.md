## 📊 HALLAZGOS CLAVE - EDA

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