import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import nidaqmx
from nidaqmx.constants import AcquisitionType
from threading import Thread, Event
from collections import deque
import datetime
import time

fs = 10000            
canal = "Dev2/ai0"    
tamano_bloque = int(fs * 0.05)  
ventana_tiempo = 3.0             

buffer_graf = deque(maxlen=int(fs * ventana_tiempo))  # solo últimos 3 s
datos_guardados = []  # toda la señal

adquiriendo = Event()
detener_hilo = Event()
thread_lectura = None

def hilo_lectura():
    """Lee continuamente datos del DAQ en un hilo aparte."""
    global datos_guardados, buffer_graf
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(canal)
    task.timing.cfg_samp_clk_timing(rate=fs, sample_mode=AcquisitionType.CONTINUOUS)
    task.start()
    print(f"\n▶ Adquisición iniciada en {canal} ({fs} Hz).")

    while not detener_hilo.is_set():
        if adquiriendo.is_set():
            try:
                datos = task.read(number_of_samples_per_channel=tamano_bloque)
                buffer_graf.extend(datos)
                datos_guardados.extend(datos)
            except Exception as e:
                print("⚠ Error de lectura:", e)
                break
        else:
            time.sleep(0.05)

    task.stop()
    task.close()
    print("Adquisición detenida")

def iniciar(event):
    global thread_lectura
    if not adquiriendo.is_set():
        if thread_lectura is None or not thread_lectura.is_alive():
            detener_hilo.clear()
            thread_lectura = Thread(target=hilo_lectura, daemon=True)
            thread_lectura.start()
        adquiriendo.set()
        print("▶ Grabando...")

def detener(event):
    """Detiene y guarda los datos."""
    adquiriendo.clear()
    detener_hilo.set()
    time.sleep(0.3)

    if datos_guardados:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"senal_intentoproyecto{timestamp}.txt"
        tiempos = np.arange(len(datos_guardados)) / fs
        data = np.column_stack((tiempos, datos_guardados))
        np.savetxt(nombre_archivo, data, fmt="%.6f", header="Tiempo(s)\tVoltaje(V)")
        print(f"Señal guardada en {nombre_archivo} ({len(datos_guardados)} muestras)")
    else:
        print("No se capturaron datos")


fig, ax = plt.subplots(figsize=(10, 4))
plt.subplots_adjust(bottom=0.25)
linea, = ax.plot([], [], lw=1.2, color='royalblue')
ax.set_xlim(0, ventana_tiempo)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Voltaje [V]")
ax.set_title("Señal EMG")
ax.grid(True, linestyle="--", alpha=0.6)

x = np.linspace(0, ventana_tiempo, int(fs * ventana_tiempo))
y = np.zeros_like(x)

def actualizar(frame):
    if len(buffer_graf) > 0:
        y = np.array(buffer_graf)
        if len(y) < len(x):
            y = np.pad(y, (len(x)-len(y), 0), constant_values=0)
        linea.set_data(x, y)
    return linea,


ax_iniciar = plt.axes([0.3, 0.1, 0.15, 0.075])
ax_detener = plt.axes([0.55, 0.1, 0.2, 0.075])
btn_iniciar = Button(ax_iniciar, 'Iniciar', color='purple', hovercolor='green')
btn_detener = Button(ax_detener, 'Detener y Guardar', color='pink', hovercolor='red')
btn_iniciar.on_clicked(iniciar)
btn_detener.on_clicked(detener)


from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, actualizar, interval=50, blit=True)
plt.tight_layout()
plt.show()