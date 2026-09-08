# Guía de la demostración en vivo

**Clase 5 — Automatización del Ciclo de Vida de Modelos de IA · Bloque 5.2 · 6 minutos**

Los estudiantes solo observan. En ningún momento se les pide escribir código.

---

## Antes de la clase (15 minutos, una sola vez)

1. Crear un repositorio **público** en tu cuenta de GitHub (público hace que Actions sea gratis y que los estudiantes puedan mirarlo después).
2. Subir el contenido de esta carpeta.
3. Ir a la pestaña **Actions** y habilitar los workflows si GitHub lo pide.
4. Hacer un push cualquiera para que corra por primera vez. **Debe salir todo en verde.**
5. Dejar dos pestañas abiertas en el navegador: el archivo `src/config.py` en modo edición y la pestaña Actions.

> **Verificación obligatoria:** si la primera corrida no sale verde, no hagas la demo en vivo. Usa el video de respaldo.

---

## Guion de los 6 minutos

### Minuto 0 — Encuadre (30 segundos)

> "Voy a hacer un cambio de una línea en la configuración de un modelo de scoring. No voy a tocar nada más. Miren lo que pasa."

Mostrar la pestaña Actions con la última corrida en verde. Que vean el estado normal antes de romperlo.

### Minuto 1 — El cambio (1 minuto)

Abrir `src/config.py` **directamente en la interfaz web de GitHub** (botón del lápiz). No usar el editor local: la edición en el navegador es más visible y evita el ruido de la terminal.

Cambiar una línea:

```python
UMBRAL_DECISION = 0.50    # antes
UMBRAL_DECISION = 0.60    # después
```

Mientras se hace, narrar:

> "El área de Riesgos pide ser más exigente. Subir el umbral significa aprobar solo a los solicitantes con mejor score. Parece prudente. Nadie diría que esto es peligroso."

Hacer **Commit changes** directo a `main`.

### Minuto 2 — Arranca solo (1 minuto)

Ir a Actions. La corrida ya está en marcha sin que nadie la haya lanzado.

> "Nadie apretó ningún botón. El pipeline se disparó con el commit."

Ir mostrando cómo pasan los jobs 1 y 2 en verde:

- **Pruebas de código y datos** — verde. El cambio no rompió nada técnicamente.
- **Entrenamiento** — verde. El modelo entrenó perfectamente.

> "Fíjense: hasta acá todo bien. El código funciona. El modelo entrenó. En la mayoría de las organizaciones, esto ya estaría en producción."

### Minuto 4 — El rechazo (2 minutos)

El job **3 · Puertas de calidad** se pone en rojo. Abrirlo y mostrar la salida:

```
  PUERTAS DE CALIDAD  ·  umbral de decision = 0.60
  [PASA ]  AUC (capacidad de ordenar el riesgo)   0.7461   (>= 0.70)
  [PASA ]  Recall de incumplidores                0.8729   (>= 0.65)
  [FALLA]  Tasa de aprobacion                     0.3900   (entre 0.42 y 0.62)
  [PASA ]  Brecha entre segmentos                 0.0861   (<= 0.15)
  VEREDICTO: MODELO RECHAZADO  ·  despliegue bloqueado
```

**Los tres puntos que hay que decir aquí, en este orden:**

1. **El AUC no se movió ni un decimal.** Es exactamente 0.7461 en las dos corridas. La métrica que casi todos usan para decidir si un modelo es bueno no detectó absolutamente nada, porque el AUC mide la capacidad de ordenar el riesgo y no depende del umbral. *"Si su puerta de calidad fuera solo AUC, esto pasaba."*

2. **El modelo mejoró en lo técnico y empeoró en lo comercial.** El recall de incumplidores subió de 0.81 a 0.87: detecta más morosos. Pero la tasa de aprobación cayó a 39%, por debajo del 42% que el negocio tolera. *No hay un modelo "mejor": hay un intercambio, y alguien tuvo que decidir de antemano cuál es aceptable.*

3. **La regla estaba escrita antes.** El comité definió la banda 42%–62% en frío, sin saber qué modelo vendría. Por eso esto es una puerta de calidad y no una negociación.

### Minuto 5 — El salto en gris (30 segundos)

Mostrar el job **4 · Despliegue a producción** en gris, con el ícono de omitido.

> "El despliegue nunca ocurrió. Y quiero que noten algo: no hubo ninguna reunión, ningún correo, ninguna firma. El sistema rechazó el modelo solo, en noventa segundos, un martes a las ocho de la noche."

### Minuto 6 — Revertir (30 segundos)

Volver `UMBRAL_DECISION` a `0.50` y hacer commit. Dejar la corrida en marcha de fondo mientras se sigue con la clase; a los pocos minutos estará en verde otra vez.

> "Revertir también es automático. Si revertir es difícil, desplegar rápido es imposible."

---

## Preguntas que suelen aparecer

**"¿Y si el umbral correcto sí fuera 0.60?"**
Entonces el comité tiene que cambiar la banda de aprobación, con acta, y después desplegar. El punto no es que 0.60 esté mal: es que la decisión de aflojar la puerta sea explícita, trazable y de alguien con nombre, en vez de colarse dentro de un cambio técnico.

**"¿Esto no frena al equipo?"**
Al revés. Lo que frena es enterarse tres meses después por Cobranzas. El pipeline dio la respuesta en noventa segundos.

**"¿Quién define esos umbrales?"**
Negocio y Riesgos, no el equipo de datos. Es exactamente el trabajo que van a hacer en el taller de hoy, y el tema de gobierno de la clase 6.

**"¿Por qué la brecha entre segmentos aparece si nadie la pidió?"**
Porque el 61% de la cartera son mujeres emprendedoras y están sobrerrepresentadas en el canal digital. Un modelo puede mejorar en promedio y empeorar en un segmento. Es la pregunta que el comité de riesgos les hará en el taller.

---

## Plan B

Si Actions falla, se cae la red o el runner se demora:

- Tener grabado un video de 3 minutos con la corrida completa (grabarlo el día anterior, cuando se hace la verificación previa).
- Alternativa mínima: correr todo local en la terminal con `python -m src.evaluar` para los dos umbrales. Pierde el efecto del "se disparó solo", pero conserva el del rechazo automático.

---

## Ejecución local

```bash
pip install -r requirements.txt

pytest tests/test_codigo.py tests/test_datos.py -v   # capas 1 y 2
python -m src.datos                                  # contrato de datos
python -m src.entrenar                               # entrenamiento
pytest tests/test_modelo.py -v                       # capa 3
python -m src.evaluar                                # puertas de calidad
echo $?                                              # 0 = aprobado · 1 = rechazado
```
