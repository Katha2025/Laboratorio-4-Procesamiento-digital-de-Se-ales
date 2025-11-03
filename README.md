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

**Imagen de la señal adquirida**


![Screenshot_20251101_193611_Chrome](https://github.com/user-attachments/assets/a0021c15-a75b-45f8-b910-36b59f152fd2)


**Segmentación de la señal en 5 contracciones**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs = 2000  # Hz

senal_proc = np.abs(senal - np.mean(senal))

#filtro pasabajos
b, a = butter(2, 2 / (fs / 2), btype='low')
filtrada = filtfilt(b, a, senal_proc)

#normalizar
filtrada_norm = filtrada / np.max(filtrada)

#deteccion de umbrales, cuando pasa el umbral inicia una contraccion y cuando vuelve a estar debajo del umbral se termina la contraccion
umbral = np.mean(filtrada_norm) + 0.5 * np.std(filtrada_norm)
en_contraccion = filtrada_norm > umbral
inicios = np.where(np.diff(en_contraccion.astype(int)) == 1)[0]
fines = np.where(np.diff(en_contraccion.astype(int)) == -1)[0]

if len(fines) < len(inicios):
    fines = np.append(fines, len(senal) - 1)
elif len(fines) > len(inicios):
    fines = fines[:len(inicios)]

print(f"Contracciones detectadas: {len(inicios)}")

plt.figure(figsize=(12, 4))
plt.plot(t, senal / np.max(np.abs(senal)), label="Señal EMG", color='hotpink', alpha=0.7)
plt.plot(t, filtrada_norm, label="Señal filtrada", color='palevioletred', linewidth=2)

for ini, fin in zip(inicios, fines):
    plt.axvspan(t[ini], t[fin], color='skyblue', alpha=0.3)

plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.title('Detección y segmentación de contracciones musculares simuladas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

segmentos = [senal[ini:fin] for ini, fin in zip(inicios, fines)]
print(f"Se extrajeron {len(segmentos)} segmentos.")
```

<img width="1205" height="437" alt="image" src="https://github.com/user-attachments/assets/4cc19c3c-bc6b-4665-896d-753ffca77649" />

Ahora se calculó la media y la mediana de las frecuencias y se tabularon por cada contracción.

**Código del cáculo de la media y mediana**

``` phyton
import numpy as np

def analizar_segmento(segmento, fs):
    N = len(segmento)
    if N < 2:
        return 0, 0

    ventana = np.hanning(N)
    fft_vals = np.fft.fft(segmento * ventana)
    freqs = np.fft.fftfreq(N, 1/fs)
    freqs = freqs[:N//2]
    potencia = np.abs(fft_vals[:N//2])**2

    potencia[0] = 0

    if np.sum(potencia) < 1e-12:
        return 0, 0

    f_mean = np.sum(freqs * potencia) / np.sum(potencia)

    potencia_acum = np.cumsum(potencia)
    mitad_potencia = potencia_acum[-1] / 2
    idx_median = np.where(potencia_acum >= mitad_potencia)[0]
    f_median = freqs[idx_median[0]] if len(idx_median) > 0 else 0

    return f_mean, f_median
```
Se obtuvieron como resultado las siguientes gráficas:
```phyton
for i, (ini, fin) in enumerate(zip(inicios, fines), start=1):
    t_seg = t[ini:fin]
    senal_seg = senal[ini:fin]

    plt.figure(figsize=(6, 2))
    plt.plot(t_seg, senal_seg, color='mediumvioletred')
    plt.title(f"Contracción {i}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (V)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
```
<img width="613" height="763" alt="image" src="https://github.com/user-attachments/assets/d56db413-a4aa-4bd4-9727-b35ad457ee00" />

Y por último la tabulación de la media y mediana.

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

<img width="530" height="128" alt="image" src="https://github.com/user-attachments/assets/746e734a-ba06-42fb-afce-59c3d07b86a7" />


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

<img width="701" height="400" alt="image" src="https://github.com/user-attachments/assets/5664bf75-0de4-410d-8c15-b5e8419044ee" />


<img width="798" height="389" alt="image" src="https://github.com/user-attachments/assets/d14b6b65-effe-4c17-86e9-d85ad07c079a" />


**Análisis de la variación de las frecuencias a lo largo de las contracciones simuladas**

Durante las cinco contracciones simuladas, se logra obervar que tanto la frecuencia media como la frecuencia mediana se mantienen prácticamente estables al inicio, pero al llegar a la última contracción ambas caen, en las primeras cuatro contracciones, la frecuencia media ronda los 21 Hz y la mediana cerca de los 8.5 Hz, lo que indica una señal de EMG constante y sin signos de cansancio. Sin embargo, en la quinta contracción se observa una disminución donde la media baja a 15.35 Hz y la mediana a 4.38 Hz, lo que refleja la aparición de la fatiga. Esta caída en las frecuencias representa cómo, al fatigarse el músculo, la señal pierde potencia y las fibras ya no se activan con la misma eficiencia, simulando el comportamiento real de una contracción muscular prolongada.

<img width="446" height="897" alt="image" src="https://github.com/user-attachments/assets/8b5a19d6-a3be-40f5-a56b-e4a18de1c9eb" />

# Parte B

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


El paso a seguir fue calcular para cada contracción la media y la mediana de la frecuencia.

``` phyton

import numpy as np

def analizar_segmento2(segmento, fs):
    N = len(segmento)
    if N < 2:
        return 0, 0

    ventana = np.hanning(N)
    fft_vals = np.fft.fft(segmento * ventana)
    freqs = np.fft.fftfreq(N, 1/fs)
    freqs = freqs[:N//2]
    potencia = np.abs(fft_vals[:N//2])**2

    potencia[0] = 0

    if np.sum(potencia) < 1e-12:
        return 0, 0

    f_mean2 = np.sum(freqs * potencia) / np.sum(potencia)

    potencia_acum = np.cumsum(potencia)
    mitad_potencia = potencia_acum[-1] / 2
    idx_median = np.where(potencia_acum >= mitad_potencia)[0]
    f_median2 = freqs[idx_median[0]] if len(idx_median) > 0 else 0

    return f_mean2, f_median2

```

``` phyton

resultados2 = []
for i, seg in enumerate(segmentos2):
    f_mean2, f_median2 = analizar_segmento2(seg, fs)
    resultados2.append((i+1, f_mean2, f_median2))

print("Contracción | Frecuencia media (Hz) | Frecuencia mediana (Hz)")
for r in resultados2:
    print(f"{r[0]:11.2f} | {r[1]:21.2f} | {r[2]:21.2f}")

```
Contracción | Frecuencia media (Hz) | Frecuencia mediana (Hz)


       1.00 |                 18.47 |                 18.02
       
       
       2.00 |                 26.15 |                 25.97

       
       3.00 |                 19.97 |                 19.80

       
       4.00 |                 18.24 |                 17.86

       
       5.00 |                201.61 |                200.00

       
       6.00 |                 35.28 |                 35.09

       
       7.00 |                 16.36 |                 16.26

       
       8.00 |                 15.06 |                 14.81

       
       9.00 |                 17.89 |                 17.70

       
      10.00 |                 15.13 |                 14.60

      
      11.00 |                 20.24 |                 20.00

      
      12.00 |                 30.23 |                 29.85

      
      13.00 |                 44.81 |                 44.44

      
      14.00 |                 20.77 |                 20.20

      
      15.00 |                 14.52 |                 14.18

      
      16.00 |                 18.62 |                 18.35
      
      17.00 |                 28.35 |                 28.17
      
      18.00 |                201.60 |                200.00
      
      19.00 |                 38.56 |                 38.46
      
      20.00 |                 50.11 |                 50.00
      
      21.00 |                 24.57 |                 24.39
      
      22.00 |                 18.00 |                 17.86
      
      
      23.00 |                 17.03 |                 16.81
      
      24.00 |                 17.50 |                 17.39
      
      25.00 |                 15.36 |                 14.18
      
      26.00 |                 15.87 |                 12.58
      
      27.00 |                 15.91 |                 15.50
      
      28.00 |                 19.34 |                 19.05
      
      29.00 |                 27.14 |                 27.03
      
      30.00 |                 27.64 |                 27.40
      
      31.00 |                 47.68 |                 47.62
      
      32.00 |                 30.37 |                 30.30
      
      33.00 |                 64.61 |                 64.52
      
      34.00 |                 15.67 |                 15.50
      
      35.00 |                 15.71 |                 15.38
      
      36.00 |                 20.71 |                 20.41
      
      37.00 |                201.77 |                200.00
      
      38.00 |                 20.19 |                 19.80
      
      39.00 |                 16.19 |                 16.00
      
      40.00 |                 21.56 |                 21.28
      
      41.00 |                 31.01 |                 30.77
      
      42.00 |                167.66 |                166.67
      
      43.00 |                 71.57 |                 71.43
      
      44.00 |                 48.97 |                 48.78
      
      45.00 |                 31.34 |                 31.25
      
      46.00 |                 22.01 |                 21.98
      
      47.00 |                 24.88 |                 24.69
      
      48.00 |                 18.66 |                 18.52
      
      49.00 |                 30.76 |                 30.30
      
      50.00 |                 28.75 |                 28.17
      
      51.00 |                 83.81 |                 83.33
      
      52.00 |                 23.37 |                 22.99
      
      53.00 |                 13.51 |                 13.25
      
      54.00 |                 13.06 |                 12.12
      
      55.00 |                 35.12 |                 35.09
      
      56.00 |                133.88 |                133.33
      
      57.00 |                 20.83 |                 20.41
      
      58.00 |                 26.24 |                 25.97
      
      59.00 |                 15.35 |                 14.71
      
      60.00 |                 64.84 |                 64.52
      
      61.00 |                 16.73 |                 16.53
      
      62.00 |                 16.84 |                 16.26
      
      63.00 |                 17.32 |                 17.24
      
      64.00 |                 19.00 |                 18.87
      
      65.00 |                 29.91 |                 29.85
      
      66.00 |                 29.09 |                 28.99
      
      67.00 |                 32.89 |                 32.79
      
      68.00 |                 24.20 |                 23.81
      
      69.00 |                 42.05 |                 41.67
      
      70.00 |                 18.83 |                 17.86
      
      71.00 |                 21.03 |                 19.80
      
      72.00 |                 20.05 |                 19.42
      
      73.00 |                 19.06 |                 18.87
      
      74.00 |                 44.51 |                 44.44
      
      75.00 |                 83.59 |                 83.33
      
      76.00 |                 40.34 |                 40.00
      
      77.00 |                 40.14 |                 40.00
      
      78.00 |                 31.45 |                 31.25
      
      79.00 |                 39.50 |                 39.22
      
      80.00 |                 16.50 |                 16.26
      
      81.00 |                 14.17 |                 13.89
      
      82.00 |                 18.11 |                 17.86
      
      83.00 |                 23.89 |                 23.81
      
      84.00 |                 39.49 |                 39.22
      
      85.00 |                 24.68 |                 24.39
      
      86.00 |                 20.71 |                 20.20
      
      87.00 |                 11.65 |                  9.43
      
      88.00 |                 29.97 |                 29.85
      
      89.00 |                 23.07 |                 22.73
      
      90.00 |                 37.84 |                 37.74
      
      91.00 |                 40.95 |                 40.82
      
      92.00 |                 69.16 |                 68.97
      
      93.00 |                 25.42 |                 25.32
      
      94.00 |                 26.40 |                 25.97
      
      95.00 |                 26.33 |                 25.97
      
      96.00 |                 37.91 |                 37.74
      
      97.00 |                 24.03 |                 23.81
      
      98.00 |                 34.13 |                 33.90
      
      99.00 |                 50.08 |                 50.00
     
     100.00 |                 42.62 |                 42.55
     
     101.00 |                167.63 |                166.67
     
     102.00 |                 24.67 |                 24.10
     
     103.00 |                 80.34 |                 80.00
     
     104.00 |                 28.96 |                 28.57
     105.00 |                 24.10 |                 23.81
     
     106.00 |                 37.45 |                 37.04
     
     107.00 |                 15.47 |                 15.38
     
     108.00 |                 19.46 |                 19.05
     
     109.00 |                 44.53 |                 44.44
     
     110.00 |                 27.15 |                 27.03
     
     111.00 |                 19.39 |                 19.23
     
     112.00 |                 22.03 |                 21.74
     
     113.00 |                 55.77 |                 55.56
     
     114.00 |                 40.24 |                 40.00
     
     115.00 |                 23.08 |                 22.73
     
     116.00 |                 21.19 |                 20.83
     
     117.00 |                 87.41 |                 86.96
     
     118.00 |                 43.78 |                 43.48
     
     119.00 |                  0.00 |                  0.00
     
     120.00 |                 83.48 |                 83.33
     
     121.00 |                133.95 |                133.33
     
     122.00 |                 19.40 |                 19.23
     
     123.00 |                 59.18 |                 58.82
     
     124.00 |                 25.89 |                 25.32
     
     125.00 |                 15.21 |                 15.04
     
     126.00 |                 16.22 |                 15.87
     
     127.00 |                 20.08 |                 20.00
     
     128.00 |                 50.08 |                 50.00
     
     129.00 |                 35.83 |                 35.71
     
     130.00 |                 74.28 |                 74.07
     
     131.00 |                 59.10 |                 58.82
     
     132.00 |                 21.43 |                 21.05
     
     133.00 |                 29.28 |                 28.99
     
     134.00 |                 20.32 |                 20.00
     
     135.00 |                 38.04 |                 37.74
     
     136.00 |                 19.27 |                 18.87
     
     137.00 |                 22.54 |                 22.22
     
     138.00 |                 48.97 |                 48.78
     
     139.00 |                 24.27 |                 24.10
     
     140.00 |                 13.31 |                 13.16
     
     141.00 |                 18.53 |                 18.02
     
     142.00 |                 20.06 |                 19.80
     
     143.00 |                 17.92 |                 17.70
     
     144.00 |                 24.35 |                 24.10
     
     145.00 |                 47.80 |                 47.62
     
     146.00 |                 40.16 |                 40.00
     
     147.00 |                 64.65 |                 64.52
     
     148.00 |                 34.58 |                 34.48
     
     149.00 |                 80.21 |                 80.00
     
     150.00 |                 25.45 |                 25.00
     
     151.00 |                 27.96 |                 27.78
     
     152.00 |                 43.79 |                 43.48
     
     153.00 |                 24.42 |                 24.10
     
     154.00 |                 19.64 |                 19.42
     
     155.00 |                 16.84 |                 16.39
     
     156.00 |                 25.44 |                 25.00
     
     157.00 |                 54.12 |                 54.05
     
     158.00 |                 30.90 |                 30.77
     
     159.00 |                 29.49 |                 29.41
     
     160.00 |                 24.90 |                 24.69
     
     161.00 |                 24.37 |                 24.10
     
     162.00 |                 33.71 |                 33.33
     
     163.00 |                 17.92 |                 17.70
     
     164.00 |                 21.29 |                 20.83
     
     165.00 |                 29.84 |                 29.41
     
     166.00 |                 17.35 |                 17.09
     
     167.00 |                 41.85 |                 41.67
     
     168.00 |                 40.92 |                 40.82
     
     169.00 |                 19.60 |                 19.42
     
     170.00 |                 35.16 |                 35.09
     
     171.00 |                 43.63 |                 43.48
     
     172.00 |                 25.32 |                 25.00
     
     173.00 |                 18.65 |                 18.35
     
     174.00 |                 19.56 |                 19.23
     
     175.00 |                 23.98 |                 23.26
     
     176.00 |                 15.76 |                 15.50
     
     177.00 |                 23.35 |                 22.99
     
     178.00 |                 21.99 |                 21.98
     
     179.00 |                 22.76 |                 22.73
     
     180.00 |                 38.51 |                 38.46
     
     181.00 |                 33.95 |                 33.90
     
     182.00 |                 66.92 |                 66.67
     
     183.00 |                 33.53 |                 33.33
     
     184.00 |                 19.60 |                 19.23
     
     185.00 |                 71.91 |                 71.43
     
     186.00 |                 18.45 |                 18.18
     
     187.00 |                 35.44 |                 35.09
     
     188.00 |                 25.44 |                 25.00
     
     189.00 |                 19.89 |                 19.61
     190.00 |                 19.54 |                 19.42
     
     191.00 |                 64.65 |                 64.52
     
     192.00 |                105.51 |                105.26
     193.00 |                 23.96 |                 23.81
     
     194.00 |                 24.23 |                 23.81
     
     195.00 |                 20.32 |                 20.00
     
     196.00 |                 32.99 |                 32.79
     
     197.00 |                 30.94 |                 30.77
     
     198.00 |                 47.89 |                 47.62
     
     199.00 |                 17.00 |                 16.81
     
     200.00 |                 21.20 |                 20.83
     
     201.00 |                 22.39 |                 22.22
     
     202.00 |                 23.88 |                 23.81
     
     203.00 |                 28.35 |                 28.17
     
     204.00 |                 23.58 |                 23.26
     
     205.00 |                 23.33 |                 23.26
     
     206.00 |                 34.97 |                 34.48
     
     207.00 |                 23.20 |                 22.73
     
     208.00 |                 31.85 |                 31.25
     
     209.00 |                 39.61 |                 39.22
     
     210.00 |                 51.44 |                 51.28
     
     211.00 |                 28.91 |                 28.57
     
     212.00 |                 34.61 |                 34.48
     
     213.00 |                 30.02 |                 29.85
     
     214.00 |                 21.34 |                 21.28
     
     215.00 |                100.23 |                100.00
     
     216.00 |                183.19 |                181.82
     
     217.00 |                 30.22 |                 29.85
     
     218.00 |                 28.75 |                 28.57
     
     219.00 |                 38.16 |                 37.74
     
     220.00 |                 42.91 |                 42.55
     
     221.00 |                 29.66 |                 29.41
     
     222.00 |                 17.37 |                 17.09
     
     223.00 |                 14.63 |                 14.39
     
     224.00 |                 44.71 |                 44.44
     
     225.00 |                 35.26 |                 35.09
     
     226.00 |                125.43 |                125.00
     
     227.00 |                 25.81 |                 25.64
     
     228.00 |                 43.67 |                 43.48
     
     229.00 |                 20.09 |                 19.80
     
     230.00 |                 27.61 |                 27.03
     
     231.00 |                 32.89 |                 32.26
     
     232.00 |                 36.87 |                 36.36
     
     233.00 |                 24.35 |                 23.81
     
     234.00 |                 15.48 |                 14.18
     
     235.00 |                 52.77 |                 52.63
     
     236.00 |                 35.94 |                 35.71
     
     237.00 |                 77.09 |                 76.92
     
     238.00 |                 35.21 |                 35.09
     
     239.00 |                 33.97 |                 33.90
     
     240.00 |                 20.77 |                 20.20
     
     241.00 |                 34.33 |                 33.90
     
     242.00 |                 53.17 |                 52.63
     
     243.00 |                 28.15 |                 27.78
     244.00 |                 12.20 |                 12.05
     
     245.00 |                 16.73 |                 16.67
     
     246.00 |                 23.39 |                 22.99
     247.00 |                 24.82 |                 24.69
     
     248.00 |                 16.22 |                 16.13
     
     249.00 |                 24.49 |                 24.39
     250.00 |                167.63 |                166.67
     
     251.00 |                 38.15 |                 37.74
     
     252.00 |                 31.63 |                 31.25
     
     253.00 |                 41.26 |                 40.82
     
     254.00 |                 27.45 |                 27.03
     
     255.00 |                 32.96 |                 32.26
     
     256.00 |                 13.75 |                 12.90
     
     257.00 |                 17.56 |                 17.24
     
     258.00 |                 17.47 |                 17.24
     
     259.00 |                 26.89 |                 26.67
     
     260.00 |                 34.68 |                 34.48
     
     261.00 |                 83.49 |                 83.33
     
     262.00 |                 55.61 |                 55.56
     
     263.00 |                 37.33 |                 37.04
     
     264.00 |                 33.22 |                 32.79
     
     265.00 |                 21.79 |                 21.51
     
     266.00 |                 59.11 |                 58.82
     
     267.00 |                 19.47 |                 19.23
     
     268.00 |                 23.55 |                 22.99
     
     269.00 |                 15.74 |                 15.38
     
     270.00 |                 30.42 |                 30.30
     
     271.00 |                 26.03 |                 25.97
     
     272.00 |                 42.01 |                 41.67
     
     273.00 |                 23.30 |                 22.99
     
     274.00 |                 26.25 |                 25.97
     
     275.00 |                 50.51 |                 50.00
     
     276.00 |                 21.77 |                 21.05
     
     277.00 |                 24.40 |                 24.10
     
     278.00 |                 23.83 |                 23.53
     
     279.00 |                 32.07 |                 31.75
    
     280.00 |                 17.09 |                 16.81
     
     281.00 |                 18.54 |                 18.35
     
     282.00 |                 95.53 |                 95.24
     
     283.00 |                 29.16 |                 28.99
     
     284.00 |                 46.60 |                 46.51
    
     285.00 |                 41.96 |                 41.67
     
     286.00 |                 83.49 |                 83.33
     
     287.00 |                 71.57 |                 71.43
     
     288.00 |                 57.29 |                 57.14
     
     289.00 |                 17.37 |                 17.09
     
     290.00 |                 12.06 |                 11.43
     
     291.00 |                 15.56 |                 13.33
     
     292.00 |                 26.50 |                 25.97
     
     293.00 |                 22.67 |                 22.22
     
     294.00 |                 18.02 |                 17.70
     
     295.00 |                 21.15 |                 20.83
     
     296.00 |                 17.70 |                 17.39
     
     297.00 |                 26.83 |                 26.67
     
     298.00 |                 41.78 |                 41.67
     
     299.00 |                 32.87 |                 32.79
     
     300.00 |                 21.06 |                 21.05
     
     301.00 |                 21.35 |                 20.83
     
     302.00 |                 83.70 |                 83.33
     
     303.00 |                 18.16 |                 17.86
     
     304.00 |                 13.62 |                 13.16
     
     305.00 |                 17.73 |                 17.39
     
     306.00 |                 18.54 |                 18.18
     
     307.00 |                 23.72 |                 23.53
     
     308.00 |                 31.86 |                 31.75
     
     309.00 |                 74.25 |                 74.07
     
     310.00 |                 33.42 |                 33.33
     
     311.00 |                 31.40 |                 31.25
     
     312.00 |                 19.30 |                 19.23
     
     313.00 |                 13.08 |                 13.07
     
     314.00 |                 18.56 |                 18.52
     
     315.00 |                 23.04 |                 22.99
     
     316.00 |                 34.00 |                 33.90
     
     317.00 |                 26.12 |                 25.97
     
     318.00 |                 20.26 |                 20.20
     
     
     319.00 |                 32.81 |                 32.79
     
     320.00 |                 35.85 |                 35.71
     
     321.00 |                 34.57 |                 34.48
     
     322.00 |                 46.62 |                 46.51
     
     323.00 |                 34.59 |                 34.48
     
     324.00 |                 29.05 |                 28.99
     
     325.00 |                 55.62 |                 55.56
     
     326.00 |                 52.82 |                 52.63
     
     327.00 |                 20.00 |                 19.61
     
     328.00 |                 17.42 |                 17.09
     
     329.00 |                 26.63 |                 26.32
     
     330.00 |                 30.58 |                 29.85
     
     331.00 |                 33.76 |                 33.33
     
     332.00 |                111.70 |                111.11
     
     333.00 |                 15.32 |                 15.15
     
     334.00 |                 13.67 |                 13.42
     
     335.00 |                 40.90 |                 40.82
     
     336.00 |                 36.39 |                 36.36
     
     337.00 |                 50.12 |                 50.00
     
     338.00 |                 19.59 |                 19.23
     
     339.00 |                 23.93 |                 23.81
     
     340.00 |                 29.10 |                 28.57
     
     341.00 |                 36.81 |                 36.36
     
     342.00 |                 34.11 |                 33.90
     
     343.00 |                 28.34 |                 27.78
     
     344.00 |                 34.23 |                 33.90
     
     345.00 |                 29.23 |                 28.57
     
     346.00 |                 18.21 |                 18.18
     
     347.00 |                 15.15 |                 14.71
     
     348.00 |                 25.41 |                 25.32
     
     349.00 |                 25.17 |                 25.00
     
     350.00 |                400.00 |                400.00
     
     351.00 |                 66.90 |                 66.67
     
     352.00 |                 19.77 |                 19.61
     
     353.00 |                 16.70 |                 14.29
     
     354.00 |                 20.56 |                 19.80
     
     355.00 |                 26.97 |                 26.67
     
     356.00 |                 54.24 |                 54.05
     
     357.00 |                 22.69 |                 22.22
     
     358.00 |                 18.80 |                 18.35
     
     359.00 |                118.43 |                117.65
     
     360.00 |                 13.43 |                 12.90
     
     361.00 |                 17.77 |                 17.54
     
     362.00 |                 27.04 |                 27.03
     
     363.00 |                 26.47 |                 26.32
     
     364.00 |                 69.13 |                 68.97
     
     365.00 |                 23.45 |                 23.26
     
     366.00 |                 38.00 |                 37.74
     
     367.00 |                 23.81 |                 23.53
     
     368.00 |                 25.35 |                 24.69
     
     369.00 |                 22.72 |                 22.47
     
     370.00 |                 33.29 |                 32.79
     
     371.00 |                 32.72 |                 32.26
     
     372.00 |                 10.18 |                  9.71
     
     373.00 |                 19.88 |                 19.61
     
     374.00 |                 32.93 |                 32.79
     
     
     375.00 |                 44.66 |                 44.44
     
     376.00 |                 25.53 |                 25.32
     
     377.00 |                 32.96 |                 32.79
     
     378.00 |                 26.56 |                 26.32
     
     379.00 |                 66.91 |                 66.67
     
     380.00 |                 27.92 |                 27.78
     
     381.00 |                 80.32 |                 80.00
     
     382.00 |                 51.58 |                 51.28
     
     383.00 |                 38.31 |                 37.74
     
     384.00 |                 45.82 |                 45.45
     
     385.00 |                 31.01 |                 30.77
     
     386.00 |                 39.56 |                 39.22
     
     387.00 |                 13.34 |                 13.25
     
     388.00 |                 21.76 |                 21.28
     
     389.00 |                 38.18 |                 37.74
     
     390.00 |                 28.05 |                 27.78
     
     391.00 |                 31.95 |                 31.75
     
     392.00 |                 50.12 |                 50.00
     
     393.00 |                 18.63 |                 18.52
     
     394.00 |                 27.18 |                 27.03
     
     395.00 |                 17.04 |                 16.95
     
     396.00 |                 12.41 |                 12.05
     
     397.00 |                 36.30 |                 35.71
     
     
     398.00 |                 28.99 |                 28.57
     
     399.00 |                 42.34 |                 41.67
     
     400.00 |                 18.36 |                 18.18
     
     401.00 |                 24.32 |                 23.81
     
     402.00 |                 15.37 |                 15.04
     
     403.00 |                 15.18 |                 14.93
     
     404.00 |                 21.87 |                 21.74
     
     405.00 |                 87.21 |                 86.96
     
     406.00 |                 28.35 |                 28.17
     
     
     407.00 |                 38.61 |                 38.46
     
     408.00 |                 20.46 |                 20.41
     
     409.00 |                 32.62 |                 32.26
     
     410.00 |                154.59 |                153.85
     
     411.00 |                 20.41 |                 20.20
     
     412.00 |                 24.11 |                 23.53
     
     413.00 |                 33.93 |                 33.90
     
     414.00 |                 31.29 |                 31.25
     
     415.00 |                 20.66 |                 20.62
     
     416.00 |                 35.11 |                 35.09
     
     417.00 |                 11.10 |                 10.36
     
     418.00 |                 16.67 |                 16.39
     
     419.00 |                 23.40 |                 22.99
     
     420.00 |                 45.13 |                 44.44
     
     421.00 |                 24.29 |                 23.81
     
     422.00 |                 38.23 |                 37.74
     
     423.00 |                 21.77 |                 21.28
     
     424.00 |                 18.56 |                 18.18
     
     425.00 |                 23.25 |                 22.99
     
     426.00 |                 31.89 |                 31.75
     
     427.00 |                500.00 |                500.00
     
     
     428.00 |                 32.81 |                 32.79
     
     429.00 |                 37.06 |                 37.04
     
     430.00 |                 18.13 |                 17.86
     
     431.00 |                 17.21 |                 16.95
     
     432.00 |                 16.79 |                 16.39
     
     433.00 |                 87.51 |                 86.96
     
     434.00 |                 22.24 |                 21.74
     
     435.00 |                 21.49 |                 21.05
     
     436.00 |                 35.11 |                 34.48
     
     437.00 |                 17.37 |                 17.24
     
     438.00 |                 43.92 |                 43.48
     
     439.00 |                 39.67 |                 39.22
     
     440.00 |                 17.42 |                 17.09
     
     441.00 |                 37.28 |                 37.04
     
     442.00 |                 22.46 |                 22.22
     
     443.00 |                 32.35 |                 32.26
     
     444.00 |                 39.33 |                 39.22
     
     445.00 |                 51.40 |                 51.28
     
     446.00 |                 28.31 |                 28.17
     
     447.00 |                 31.39 |                 31.25
     
     448.00 |                 21.19 |                 20.83
     
     449.00 |                 15.39 |                 15.27
     
     450.00 |                 40.51 |                 40.00
     
     451.00 |                 33.70 |                 33.33
     
     452.00 |                 21.02 |                 20.62
     
     453.00 |                 19.34 |                 19.05
     
     454.00 |                 27.88 |                 27.40
     
     455.00 |                 18.67 |                 18.35
     
     456.00 |                133.89 |                133.33
     
     457.00 |                 35.22 |                 35.09
     
     458.00 |                118.01 |                117.65
     
     459.00 |                339.44 |                333.33
     
     460.00 |                 31.51 |                 31.25
     
     461.00 |                 15.91 |                 15.62
     
     462.00 |                 13.75 |                 13.61
     
     463.00 |                 15.02 |                 14.39
     
     464.00 |                 18.15 |                 18.02
     
     465.00 |                 91.36 |                 90.91
     
     466.00 |                 19.12 |                 18.87
     
     467.00 |                 25.80 |                 25.32
     
     468.00 |                 19.50 |                 18.69
     
     469.00 |                 22.01 |                 21.74
     
     470.00 |                111.43 |                111.11
     
     471.00 |                 71.61 |                 71.43
     
     472.00 |                 37.90 |                 37.74
     
     473.00 |                 69.11 |                 68.97
     
     474.00 |                 20.31 |                 20.20
     
     475.00 |                 21.01 |                 20.62
     
     476.00 |                 19.09 |                 18.87
     
     477.00 |                 20.40 |                 20.00
     
     478.00 |                 32.41 |                 31.75
     
     479.00 |                 32.05 |                 31.75
     
     480.00 |                 30.48 |                 30.30
     
     481.00 |                 52.88 |                 52.63
     
     482.00 |                 43.82 |                 43.48
     
     483.00 |                 35.35 |                 35.09
     
     484.00 |                 18.11 |                 17.54
     
     485.00 |                 25.70 |                 25.32
     
     486.00 |                 35.48 |                 35.09
     
     
     487.00 |                 28.26 |                 27.78
     
     488.00 |                 23.04 |                 22.73
     
     489.00 |                 33.61 |                 33.33
     
     490.00 |                 25.80 |                 25.64
     
     491.00 |                 23.43 |                 22.99
     
     492.00 |                 25.64 |                 25.00
     
     493.00 |                 20.61 |                 20.41
     
     494.00 |                 20.39 |                 20.20
     
     495.00 |                 20.98 |                 20.41
     
     496.00 |                 77.17 |                 76.92
     
     497.00 |                 28.62 |                 28.57
     
     498.00 |                 52.71 |                 52.63
     
     499.00 |                 41.82 |                 41.67
     
     500.00 |                 14.07 |                 13.89
     
     501.00 |                 20.68 |                 20.20
     
     502.00 |                 18.78 |                 18.35
     
     503.00 |                 17.70 |                 17.39
     
     504.00 |                 22.74 |                 22.47
     
     505.00 |                 20.19 |                 20.00
     
     506.00 |                 38.99 |                 38.46
     
     507.00 |                 24.61 |                 24.10
     
     508.00 |                 17.77 |                 17.54
     
     509.00 |                 35.79 |                 35.71
     
     510.00 |                 40.93 |                 40.82
     
     511.00 |                 37.75 |                 37.74
     
     512.00 |                 27.39 |                 27.03
     
     513.00 |                 95.55 |                 95.24
     
     514.00 |                 19.24 |                 19.23
     
     515.00 |                 26.08 |                 25.97
     
     516.00 |                 21.55 |                 21.51
     
     517.00 |                 30.85 |                 30.77
     
     518.00 |                 20.67 |                 20.62
     
     519.00 |                 22.15 |                 21.98
     
     520.00 |                 17.90 |                 17.70
     
     521.00 |                 60.69 |                 60.61
     
     522.00 |                 23.86 |                 23.81
     
     523.00 |                 37.05 |                 37.04
     
     524.00 |                 15.02 |                 14.93
     
     525.00 |                 26.14 |                 25.97
     
     526.00 |                 16.15 |                 15.75
     
     527.00 |                 13.81 |                 13.70
     
     528.00 |                 21.97 |                 21.74
     
     529.00 |                 18.42 |                 18.35
     
     530.00 |                 44.53 |                 44.44
     
     531.00 |                 32.42 |                 32.26
     
     532.00 |                105.51 |                105.26
     
     533.00 |                100.33 |                100.00
     
     534.00 |                 12.18 |                 12.05
     
     535.00 |                 13.71 |                 13.25
     
     536.00 |                 13.28 |                 13.07
     
     537.00 |                 23.44 |                 22.99
     
     538.00 |                 22.80 |                 22.47
     
     539.00 |                 28.66 |                 28.17
     
     540.00 |                 17.07 |                 16.95
     
     541.00 |                 22.52 |                 22.47
     
     542.00 |                 20.09 |                 20.00
     
     543.00 |                 46.54 |                 46.51
     
     544.00 |                 21.15 |                 21.05
     
     545.00 |                 83.66 |                 83.33
     
     546.00 |                 19.27 |                 18.87
     
     547.00 |                 19.19 |                 18.87
     
     548.00 |                 13.15 |                 12.05
     
     549.00 |                 17.09 |                 16.67
     
     550.00 |                 19.67 |                 19.42
     
     551.00 |                 32.94 |                 32.79
     
     552.00 |                 25.36 |                 25.32
     
     553.00 |                 19.00 |                 18.69
     
     554.00 |                 26.90 |                 26.67
     
     555.00 |                 20.18 |                 19.80
     
     556.00 |                 15.45 |                 15.38
     
     557.00 |                 55.65 |                 55.56
     
     558.00 |                 23.69 |                 23.53
     
     559.00 |                 17.79 |                 17.70
     
     560.00 |                 40.88 |                 40.82
     
     561.00 |                111.47 |                111.11
     
     562.00 |                 27.25 |                 27.03
     
     563.00 |                 25.75 |                 25.32
     
     564.00 |                 18.22 |                 18.02
     
     565.00 |                 19.22 |                 18.87
     
     566.00 |                 36.46 |                 36.36
     
     567.00 |                 21.63 |                 21.51
     
     568.00 |                 22.53 |                 22.22
     
     569.00 |                 14.36 |                 13.89
     
     570.00 |                 12.49 |                 12.20
     
     571.00 |                 20.11 |                 19.80
     
     572.00 |                 24.47 |                 24.10
     
     573.00 |                 26.89 |                 26.67
     
     574.00 |                 29.04 |                 28.99
     
     575.00 |                 22.54 |                 22.47
     
     576.00 |                 45.51 |                 45.45
     
     577.00 |                 47.68 |                 47.62
     
     578.00 |                 31.31 |                 31.25
     
     579.00 |                 40.16 |                 40.00
     
     580.00 |                 64.66 |                 64.52
     
     581.00 |                 40.88 |                 40.82




**Gráficas de la evolución de las frecuencias**


``` phyton
from scipy.stats import linregress

resultados2 = np.array(resultados2)
contracciones2 = resultados2[:, 0]
f_media2 = resultados2[:, 1]
f_mediana2 = resultados2[:, 2]

print(f"Contracciones detectadas: {len(contracciones2)}")

slope_media, intercept_media, *_ = linregress(contracciones2, f_media2)
tendencia_media = intercept_media + slope_media * contracciones2

slope_mediana, intercept_mediana, *_ = linregress(contracciones2, f_mediana2)
tendencia_mediana = intercept_mediana + slope_mediana * contracciones2

plt.figure(figsize=(10,6))

plt.subplot(2,1,1)
plt.title("Frecuencia media")
plt.plot(contracciones2, f_media2, '*-', color='royalblue', label='Frecuencia Media')
plt.plot(contracciones2, tendencia_media, '--', color='maroon', alpha=0.8, label='Tendencia Media')
plt.xlabel("Número de Contracción")
plt.ylabel("Frecuencia (Hz)")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.title("Frecuencia mediana")
plt.plot(contracciones2, f_mediana2, '*-', color='seagreen', label='Frecuencia Mediana')
plt.plot(contracciones2, tendencia_mediana, '--', color='firebrick', alpha=0.8, label='Tendencia Mediana')
plt.xlabel("Número de Contracción")
plt.ylabel("Frecuencia (Hz)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Pendiente Frecuencia Media:   {slope_media:.4f}")
print(f"Pendiente Frecuencia Mediana: {slope_mediana:.4f}")

if slope_media < 0 and slope_mediana < 0:
    print("\nAmbas pendientes son negativas lo que indica tendencia descendente clara. Indica que hay fatiga muscular progresiva.")
elif slope_media < 0 or slope_mediana < 0:
    print("\nSolo una frecuencia desciende lo que indica posible fatiga parcial.")
else:
    print("\nNo hay tendencia descendente lo que indica que no se observa fatiga muscular evidente.")
```

<img width="1031" height="684" alt="image" src="https://github.com/user-attachments/assets/4b974645-606c-43de-ada6-70abc8d33985" />


**Discusión**

A partir de las gráficas de frecuencia media y frecuencia mediana se observa una tendencia descendente en ambas variables a lo largo del número de contracciones musculares, esta disminución  se evidencia en las pendientes negativas calculadas (−0.0026 para la frecuencia media y −0.0024 para la frecuencia mediana), lo que indica una clara pérdida de componentes de alta frecuencia en la señal EMG conforme avanza la actividad muscular. Desde el punto de vista fisiológico, este comportamiento está asociado con la fatiga muscular, ya que durante este proceso se reduce la velocidad de conducción de las fibras musculares debido a alteraciones en la concentración iónica y a la acumulación de metabolitos, como el ácido láctico. Además, el músculo tiende a reclutar un mayor número de fibras lentas (tipo I), cuyas señales presentan frecuencias más bajas. En conjunto, estos factores provocan el desplazamiento del espectro electromiográfico hacia frecuencias menores, reflejando una disminución en la capacidad contráctil y una menor eficiencia en la activación neuromuscular. Por tanto, la tendencia descendente observada en las gráficas confirma la presencia de una fatiga muscular progresiva durante las contracciones analizadas.

<img width="357" height="884" alt="image" src="https://github.com/user-attachments/assets/0375982b-114e-416f-b8f5-24c4b94798e5" />


# Parte C

 **Análisis espectral mediant**
 
Transformada Rápida de Fourier 
```python
import numpy as np
import matplotlib.pyplot as plt

fs = 10000

for i, seg in enumerate(segmentos2):
    N = len(seg)
    if N < 2:
        continue

    ventana = np.hanning(N)
    seg_ventana = seg * ventana

    fft_vals = np.fft.fft(seg_ventana)
    freqs = np.fft.fftfreq(N, 1/fs)

    fft_vals = fft_vals[:N//2]
    freqs = freqs[:N//2]
    potencia = np.abs(fft_vals)**2
   ``` 
 **Espectro de amplitud**


 ```python
 magnitud = np.abs(fft_vals) / N
plt.figure(figsize=(8,4))
plt.semilogy(freqs, magnitud, color='darkslategray')
plt.title(f"Espectro de amplitud - Contracción {i+1}")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

<img width="685" height="323" alt="image" src="https://github.com/user-attachments/assets/50949b8e-e79b-49da-92c4-931ff2caab01" />

**Comparacion de espectros**
```python
import numpy as np
import matplotlib.pyplot as plt

fs = 10000  # frecuencia de muestreo

# Ajusta aquí si tu variable real es 'segmentos_filtrados'
seg_inicio = segmentos2[0]
seg_final = segmentos2[-1]

def calcular_fft(seg, fs):
    N = len(seg)
    # Eliminar componente DC antes de la ventana
    seg = seg - np.mean(seg)
    ventana = np.hanning(N)
    seg_ventana = seg * ventana
    fft_vals = np.fft.fft(seg_ventana)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_vals = fft_vals[:N//2]
    freqs = freqs[:N//2]
    potencia = np.abs(fft_vals)**2
    return freqs, potencia

freqs_inicio, pot_inicio = calcular_fft(seg_inicio, fs)
freqs_final, pot_final = calcular_fft(seg_final, fs)

plt.figure(figsize=(10,5))
plt.semilogy(freqs_inicio, pot_inicio, label='Primeras contracciones')
plt.semilogy(freqs_final, pot_final, label='Últimas contracciones', linestyle='dashed')
plt.xlim([20, 500])  # Rango relevante para EMG
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia (escala log)')
plt.title('Comparación de espectros EMG (inicio vs. final)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
```
<img width="896" height="396" alt="image" src="https://github.com/user-attachments/assets/a412d330-7050-4c51-a8ff-81f7b324232b" />

**Identificar la reducción del contenido de alta frecuencia asociada con la fatiga 
muscular**
```python
f_high = 100  # Límite de alta frecuencia en Hz

# Para el espectro de inicio
indices_high_inicio = np.where(freqs_inicio > f_high)
contenido_high_inicio = np.sum(pot_inicio[indices_high_inicio])

# Para el espectro de final
indices_high_final = np.where(freqs_final > f_high)
contenido_high_final = np.sum(pot_final[indices_high_final])

print("Contenido alta frecuencia (inicio):", contenido_high_inicio)
print("Contenido alta frecuencia (final):", contenido_high_final)
print("Reducción relativa:", 100 * (1 - contenido_high_final / contenido_high_inicio), "%")
```
Contenido alta frecuencia (inicio): 33.368706225188234
Contenido alta frecuencia (final): 0.018871728736843073
Reducción relativa: 99.94344482938749 %
**Calcular y discutir el desplazamiento del pico espectral**

```python
segmentos_filtrados = [senal_filtrada[ini:fin] for ini, fin in zip(inicios2, fines2)]

picos = []

for seg in segmentos_filtrados:
    N = len(seg)
    if N < 2:
        picos.append(np.nan)
        continue

    seg_ventana = seg * np.hanning(N)
    fft_vals = np.fft.fft(seg_ventana)
    freqs = np.fft.fftfreq(N, 1/fs)

    fft_vals = fft_vals[:N//2]
    freqs = freqs[:N//2]
    magnitud = np.abs(fft_vals) / N

    magnitud[0] = 0

    f_pico = freqs[np.argmax(magnitud)]
    picos.append(f_pico)

picos = np.array(picos)
n = 3

pico_primeras = np.nanmean(picos[:n])
pico_ultimas = np.nanmean(picos[-n:])

desplazamiento = pico_ultimas - pico_primeras

print(f"Pico promedio primeras contracciones: {pico_primeras:.2f} Hz")
print(f"Pico promedio últimas contracciones: {pico_ultimas:.2f} Hz")
print(f"Desplazamiento del pico: {desplazamiento:.2f} Hz")
```

Pico promedio primeras contracciones: 106.32 Hz
Pico promedio últimas contracciones: 242.22 Hz
Desplazamiento del pico: 135.90 Hz
 
 
Finalmente en esta parte de los apartados se aplicó la Transformada Rápida de Fourier (FFT) a cada contracción de la señal EMG real, obteniendo los espectros de amplitud donde compararon los primeros y últimos espectros para identificar la reducción de contenido en altas frecuencias, fenómeno asociado a la fatiga muscular. También se analizó el desplazamiento del pico espectral como indicador del esfuerzo sostenido.
