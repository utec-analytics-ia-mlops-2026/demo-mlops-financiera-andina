# Pipeline MLOps — Scoring crediticio de Financiera Andina

Repositorio de demostración del curso **MLOps para escalar IA**, maestría en Management, Analytics & IA — UTEC Posgrado.

Muestra un ciclo de vida de modelo completamente automatizado: pruebas, entrenamiento, puertas de calidad y despliegue. Lo importante no es que el pipeline funcione, sino que **rechace** un modelo sin que intervenga ninguna persona.

> El caso, los datos y las cifras son ficticios. Financiera Andina no existe.

---

## Qué hace

Un modelo estima la probabilidad de incumplimiento de solicitantes de microcrédito y la convierte en un score de solvencia entre 0 y 1. Se aprueba la solicitud cuando el score supera el umbral de decisión.

Con cada `push`, el pipeline se ejecuta solo:

```
Pruebas de código  →  Pruebas de datos  →  Entrenamiento  →  Pruebas de modelo
                                                                     ↓
                       Despliegue  ←  Puertas de calidad  ←──────────┘
```

Si alguna puerta de calidad no se cumple, el job falla y **el despliegue queda omitido**.

---

## Las tres capas de pruebas

En software tradicional se prueba el código. En un sistema de IA hay que probar tres cosas distintas:

| Capa | Archivo | Qué verifica |
|---|---|---|
| **Código** | `tests/test_codigo.py` | Que las funciones hagan lo que dicen |
| **Datos** | `tests/test_datos.py` | Esquema, rangos, categorías, unidades, volumen |
| **Modelo** | `tests/test_modelo.py` | Reproducibilidad, sentido de negocio, robustez |

La prueba `test_la_unidad_monetaria_sigue_siendo_soles` ilustra el fallo clásico que solo la capa de datos detecta: si la fuente empieza a entregar los ingresos en dólares, el código sigue funcionando, el modelo entrena sin errores y las predicciones son sistemáticamente absurdas.

---

## Las puertas de calidad

Reglas escritas **antes** de entrenar, en `src/config.py`. El pipeline las evalúa solo y decide si el modelo puede ir a producción.

| Puerta | Criterio | Para qué |
|---|---|---|
| AUC | ≥ 0,70 | Capacidad de ordenar el riesgo. **No depende del umbral.** |
| Recall de incumplidores | ≥ 0,65 | Protege la morosidad de la cartera |
| Tasa de aprobación | entre 0,42 y 0,62 | Banda tolerada por el negocio |
| Brecha entre segmentos | ≤ 0,15 | Mujeres emprendedoras frente al resto |

Si el criterio de aprobación no está escrito antes, lo que hay no es una puerta de calidad: es una negociación.

---

## Estructura

```
├── .github/workflows/pipeline-mlops.yml   Los cuatro jobs encadenados
├── src/
│   ├── config.py        Umbral, puertas de calidad y contrato de datos
│   ├── datos.py         Carga, variables derivadas y validación
│   ├── entrenar.py      Entrenamiento y generación de artefactos
│   └── evaluar.py       Puertas de calidad. Sale con código 1 si alguna falla
├── tests/               Las tres capas
├── data/solicitudes.csv 6.000 solicitudes sintéticas
├── scripts/             Generador de datos (no hace falta ejecutarlo)
└── GUIA_DEMO.md         Guion de la demostración en clase
```

---

## Ejecución local

```bash
pip install -r requirements.txt

pytest tests/test_codigo.py tests/test_datos.py -v
python -m src.entrenar
pytest tests/test_modelo.py -v
python -m src.evaluar        # código de salida 0 = aprobado · 1 = rechazado
```

---

## La demostración

Cambiar una línea en `src/config.py`:

```python
UMBRAL_DECISION = 0.50   →   UMBRAL_DECISION = 0.60
```

Hacer push y observar el resultado:

| | Umbral 0,50 | Umbral 0,60 |
|---|---|---|
| AUC | 0,7461 | **0,7461 — idéntico** |
| Recall de incumplidores | 0,8093 | 0,8729 (mejora) |
| Tasa de aprobación | 0,4972 | **0,3900 — fuera de banda** |
| Brecha entre segmentos | 0,1075 | 0,0861 |
| **Veredicto** | Aprobado | **Rechazado** |

El modelo se volvió mejor detectando morosos y peor colocando créditos. El AUC —la métrica que casi todos miran— no se movió ni un decimal. La puerta de calidad es lo único que capturó el intercambio.

El guion completo está en [`GUIA_DEMO.md`](GUIA_DEMO.md).
