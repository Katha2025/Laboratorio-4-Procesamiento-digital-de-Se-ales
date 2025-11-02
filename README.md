# Laboratorio-4-Procesamiento-digital-de-Señales
# Introducción
En este laboratorio se trabajó con señales electromiográficas (EMG), las cuales reflejan la actividad eléctrica generada por los músculos durante la contracción. El objetivo principal fue aplicar técnicas de filtrado y análisis espectral para estudiar la aparición de la fatiga muscular, comparando señales emuladas y reales. Además, se emplearon herramientas de procesamiento digital para calcular la frecuencia media y mediana, y finalmente observar cómo varían con el esfuerzo sostenido.
# Parte A
En esta parte se configuró el generador de señales biológicas en modo EMG, simulando cinco contracciones musculares. La señal fue adquirida, segmentada y analizada mediante el cálculo de la frecuencia media y mediana para cada contracción. Posteriormente, se graficó la evolución de las frecuencias para identificar patrones que representaran la aparición de fatiga simulada.


**Código para la captura de la señal generada**

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

Durante el análisis se observó que la **frecuencia media** presentó un incremento en las primeras contracciones, pasando de aproximadamente **4.40 Hz** en la primera a cerca de **4.83 Hz** en las siguientes, donde se mantuvo prácticamente constante. Este comportamiento indica que, al inicio, la señal mostró un aumento en la actividad eléctrica simulada, posiblemente asociado a una mayor intensidad en las contracciones, para luego estabilizarse en un rango de frecuencia uniforme. 


# Parte B
Para poder realizar la captura de la señal se utilizaron dos electrodos de activos  sobre el músculo y un electrodo de tierra que se conectó al codo para registrar contracciones reales hasta la fatiga que fueron convertidas de los sślogo a lo digital  por un microcontrolador DAQ para finalmente ser obtenidas en el computador. La señal adquirida se filtró mediante un pasa banda , eliminando artefactos y ruido. Luego se segmentaron las contracciones, se calcularon las frecuencias media y mediana, y se analizaron sus variaciones a lo largo del tiempo, observando la disminución de la frecuencia con el aumento de la fatiga muscular.

**Código para la obtención de la señal**

``` phyton
import numpy as np
import matplotlib.pyplot as plt

emg = np.loadtxt('senal_EMG__5_mejor_mejor_mejor220251029_144237.txt', comments='#')
fs = 10000

t2= emg[:, 0]
senal2 = emg[:, 1]

inicioo = t2 <= 10
ti= t2[inicioo]
si= senal2[inicioo]

tf = t2.max()
finall = t2 >= (tf -10)
tf = t2[finall]
sf =senal2[finall]

plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.plot(ti, si, color='mediumaquamarine')
plt.title("Señal original, actividad normal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(tf, sf, color='mediumaquamarine')
plt.title("Señal original, fatiga")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.grid(True)

plt.tight_layout()
plt.show()
```

**Gráfica de la actividad del músculo sin fatiga**

![Screenshot_20251101_205002_Chrome](https://github.com/user-attachments/assets/d8cf198c-6ceb-4015-9070-39f2e56f958d)


**Gráfica de la actividad del músculo con fatiga**

![Screenshot_20251101_205006_Chrome](https://github.com/user-attachments/assets/34055b05-7b51-4b0a-94b7-033b8eb1b9b4)


Ahora bien, se aplicó un filtro pasa banda (20-450 Hz) con la finalidad de eliminar ruido y artefactos.

**Código con el filtro pasa banda**

``` phyton

from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

#filtro
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def aplicar_filtro(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

t2 = emg[:, 0]
senal2 = emg[:, 1]

senal_filtrada = aplicar_filtro(senal2, 20, 450, fs, order=4)

inicioo = t2 <= 10
ti= t2[inicioo]
si2= senal_filtrada[inicioo]

tf = t2.max()
finall = t2 >= (tf -10)
tf = t2[finall]
sf2 =senal_filtrada[finall]
plt.figure(figsize=(12, 6))

plt.subplot(2,1,1)
plt.plot(ti, si2, color='darkorchid')
plt.title("Señal filtrada, actividad normal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(tf, sf2, color='darkorchid')
plt.title("Señal filtrada, fatiga")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.grid(True)

plt.tight_layout()
plt.show()


``` 
![Screenshot_20251101_205442_Chrome](https://github.com/user-attachments/assets/5f55a01e-7e31-47e6-b988-681f8005364f)


![Screenshot_20251101_205450_Chrome](https://github.com/user-attachments/assets/ffacc1e3-15a4-4980-9187-7a9705a89ab7)


Después de se dividió la señal en el número de contracciones realizadas:

``` phyton

#deteccion de umbrales, cuando pasa el umbral inicia una contraccion y cuando vuelve a estar debajo del umbral se termina la contraccion
umbral = np.mean(senal_filtrada) + 0.5 * np.std(senal_filtrada)
contraccion2 = senal_filtrada > umbral
inicios2 = np.where(np.diff(contraccion2.astype(int)) == 1)[0]
fines2 = np.where(np.diff(contraccion2.astype(int)) == -1)[0]

if len(fines2) < len(inicios2):
    fines2 = np.append(fines2, len(senal) - 1)
elif len(fines2) > len(inicios2):
    fines2 = fines2[:len(inicios2)]

print(f"Contracciones detectadas: {len(inicios2)}")

plt.figure(figsize=(12, 4))
plt.plot(t2, senal_filtrada / np.max(np.abs(senal_filtrada)), label="Señal EMG", color='darkgoldenrod', alpha=0.7)

for ini, fin in zip(inicios2, fines2):
    plt.axvspan(t2[ini], t2[fin], color='lightpink', alpha=0.3)

plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.title('Detección y segmentación de contracciones musculares capturadas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

segmentos2 = [senal_filtrada[ini:fin] for ini, fin in zip(inicios2, fines2)]
print(f"Se extrajeron {len(segmentos2)} segmentos.")

```

![Screenshot_20251101_212030_Chrome](https://github.com/user-attachments/assets/7517d0a1-b3ea-4ee7-878a-e8961312deee)





# Parte C
Finalmente en esta parte de los apartados se aplicó la Transformada Rápida de Fourier (FFT) a cada contracción de la señal EMG real, obteniendo los espectros de amplitud donde compararon los primeros y últimos espectros para identificar la reducción de contenido en altas frecuencias, fenómeno asociado a la fatiga muscular. También se analizó el desplazamiento del pico espectral como indicador del esfuerzo sostenido.
