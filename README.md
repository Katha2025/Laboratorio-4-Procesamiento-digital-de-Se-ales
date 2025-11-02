# Laboratorio-4-Procesamiento-digital-de-Señales
# Introducción
En este laboratorio se trabajó con señales electromiográficas (EMG), las cuales reflejan la actividad eléctrica generada por los músculos durante la contracción. El objetivo principal fue aplicar técnicas de filtrado y análisis espectral para estudiar la aparición de la fatiga muscular, comparando señales emuladas y reales. Además, se emplearon herramientas de procesamiento digital para calcular la frecuencia media y mediana, y finalmente observar cómo varían con el esfuerzo sostenido.
# Parte A
En esta parte se configuró el generador de señales biológicas en modo EMG, simulando cinco contracciones musculares. La señal fue adquirida, segmentada y analizada mediante el cálculo de la frecuencia media y mediana para cada contracción. Posteriormente, se graficó la evolución de las frecuencias para identificar patrones que representaran la aparición de fatiga simulada.


**Código utilizado para el pocesamiento de la señal**

```python

import numpy as np
import matplotlib.pyplot as plt

with open('Senal_lab_4_partea_ver3.txt', 'r') as f:
    contenido = f.read()

datos = np.array(contenido.split(), dtype=float)

mitad = len(datos)//2
t = datos[:mitad]
senal = datos[mitad:]

print(f"Longitud tiempo: {len(t)} | Longitud señal: {len(senal)}")

plt.figure(figsize=(8,4))
plt.plot(t, senal, color='midnightblue')
plt.title(f"fs={1/(t[1]-t[0]):.0f}Hz, duración={t[-1]:.1f}s, muestras={len(senal)}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.grid(True)
plt.show()

```

**Imagen de la señal generada**


![Screenshot_20251101_193611_Chrome](https://github.com/user-attachments/assets/a0021c15-a75b-45f8-b910-36b59f152fd2)


Luego de la segmentación de la señal obtenida en las cinco contracciones simuladas se pasó a calcular la frecuencia media y mediana las cuales  se presentaron en una tabla y se representó la evolución de las frecuencias gráficamente.

**código del cáculo de la media y mediana**

``` phyton

resultados = []
for i, seg in enumerate(segmentos):
    f_mean, f_median = analizar_segmento(seg, fs)
    resultados.append((i+1, f_mean, f_median))

print("Contracción | Frecuencia media (Hz) | Frecuencia mediana (Hz)")
for r in resultados:
    print(f"{r[0]:11.2f} | {r[1]:21.2f} | {r[2]:21.2f}")

```
**Tablas de los resultados de la media y la mediana  de cada contracción**

![Screenshot_20251101_201516_Chrome](https://github.com/user-attachments/assets/a899e827-3706-4ea4-a516-862f1de43217)

**Gráficas de la evolución de las frecuencias**


``` phyton
resultados = np.array(resultados)
contracciones = resultados[:, 0]
f_media = resultados[:, 1]
f_mediana = resultados[:, 2]

plt.figure(figsize=(8, 4))
plt.plot(contracciones, f_media, marker='*', color='indigo')
plt.title('Frecuencia media por contracción')
plt.xlabel('Número de contracción')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8 ,4))
plt.plot(contracciones, f_mediana, marker='*', color='deeppink')
plt.title('Frecuencia mediana por contracción')
plt.xlabel('Número de contracción')
plt.ylabel('Frecuancia (Hz)')
plt.grid(True)


plt.tight_layout()
plt.show()
```

![Screenshot_20251101_202230_Chrome](https://github.com/user-attachments/assets/247abf22-d8fc-4ceb-8ba2-cddd5da6903e)

![Screenshot_20251101_202237_Chrome](https://github.com/user-attachments/assets/f874a9e3-be14-44dd-887a-e55ff27b5c09)

**Análisis de la variación de las frecuencias a lo largo de las contracciones simuladas**



# Parte B
Para poder realizar la captura de la señal se utilizaron dos electrodos de activos  sobre el músculo y un electrodo de tierra que se conectó al codo para registrar contracciones reales hasta la fatiga. La señal adquirida se filtró mediante un pasa banda de , eliminando artefactos y ruido. Luego se segmentaron las contracciones, se calcularon las frecuencias media y mediana, y se analizaron sus variaciones a lo largo del tiempo, observando la disminución de la frecuencia con el aumento de la fatiga muscular.
# Parte C
Finalmente en esta parte de los apartados se aplicó la Transformada Rápida de Fourier (FFT) a cada contracción de la señal EMG real, obteniendo los espectros de amplitud donde compararon los primeros y últimos espectros para identificar la reducción de contenido en altas frecuencias, fenómeno asociado a la fatiga muscular. También se analizó el desplazamiento del pico espectral como indicador del esfuerzo sostenido.
