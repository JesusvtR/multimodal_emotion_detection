# multimodal_emotion_detection

Repositorio del trabajo de fin de master (Universidad de Málaga) para el reconocimento de emociones multimodal.


# Troubleshooting

- Install opencv

- Video not detected
jesusg@ROS:~/catkin_ws$ ls /dev/video*
ls: cannot access '/dev/video*': No such file or directory
* Enable usb web cam and it works
jesusg@ROS:~/catkin_ws$ ls /dev/video*
/dev/video0  /dev/video1

Utilicé udevadm para obtener detalles sobre el dispositivo de video.

jesusg@ROS:~/catkin_ws$ udevadm info --name=/dev/video0 --attribute-walk

* [Instalación de `libuvc_camera`](#instalación-de-libuvc_camera)

sudo apt-get install ros-noetic-libuvc-camera

* [Solucion usb_cam image](#solución-de-problemas-comunes)

Client wants topic /usb_cam/image to have datatype...

Los topics de usb_cam son -> /usb_cam/image_raw y /usb_cam/image_raw/compressed
# Requirements

## Instalación de `pyaudio`

x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/usr/local/include -I/usr/include -I/usr/include/python3.8 -c src/pyaudio/device_api.c -o build/temp.linux-x86_64-cpython-38/src/pyaudio/device_api.o
  src/pyaudio/device_api.c:9:10: fatal error: portaudio.h: No such file or directory
      9 | #include "portaudio.h"
        |          ^~~~~~~~~~~~~
  compilation terminated.
  error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
  ----------------------------------------
  ERROR: Failed building wheel for pyaudio
Failed to build pyaudio
ERROR: Could not build wheels for pyaudio which use PEP 517 and cannot be installed directly

El error indica que portaudio.h no se encuentra, lo que significa que falta una dependencia necesaria.

* Primero, actualicé el sistema para asegurarme de que todos los paquetes estaban actualizados.

sudo apt update

* Para resolver el error de dependencia, instalé portaudio y sus dependencias.

sudo apt install portaudio19-dev python3-pyaudio

* Después de instalar las dependencias, intenté instalar pyaudio nuevamente con pip.

pip install pyaudio


# Problema con OpenCV python:
  File "/home/jesusg/.local/lib/python3.8/site-packages/albumentations/core/pydantic.py", line 15, in <module>
    cv2.INTER_NEAREST_EXACT,
AttributeError: module 'cv2' has no attribute 'INTER_NEAREST_EXACT'

* Reinstalar todas las dependencias de nuevo en este orden:

python -m pip uninstall opencv-python

python -m pip uninstall opencv-contrib-python

python -m pip install opencv-python

python -m pip install opencv-contrib-python

# Problema con librosa:
    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
TypeError: mfcc() takes 0 positional arguments but 2 positional arguments (and 1 keyword-only argument) were given

* La solucion fue instalar la version 0.9.2 de librosa

pip install librosa==0.9.2

# Ollama

Traceback (most recent call last):

  File "main.py", line 73, in <module>
    main()
  File "main.py", line 59, in main
    response = ollama.chat(model='llama3.1', messages=[
  File "/home/jesusg/.local/lib/python3.8/site-packages/ollama/_client.py", line 235, in chat
    return self._request_stream(
  File "/home/jesusg/.local/lib/python3.8/site-packages/ollama/_client.py", line 98, in _request_stream
    return self._stream(*args, **kwargs) if stream else self._request(*args, **kwargs).json()
  File "/home/jesusg/.local/lib/python3.8/site-packages/ollama/_client.py", line 69, in _request
    response = self._client.request(method, url, **kwargs)
  File "/home/jesusg/.local/lib/python3.8/site-packages/httpx/_client.py", line 827, in request
    return self.send(request, auth=auth, follow_redirects=follow_redirects)
  File "/home/jesusg/.local/lib/python3.8/site-packages/httpx/_client.py", line 914, in send
    response = self._send_handling_auth(
  File "/home/jesusg/.local/lib/python3.8/site-packages/httpx/_client.py", line 942, in _send_handling_auth
    response = self._send_handling_redirects(
  File "/home/jesusg/.local/lib/python3.8/site-packages/httpx/_client.py", line 979, in _send_handling_redirects
    response = self._send_single_request(request)
  File "/home/jesusg/.local/lib/python3.8/site-packages/httpx/_client.py", line 1015, in _send_single_request
    response = transport.handle_request(request)
  File "/home/jesusg/.local/lib/python3.8/site-packages/httpx/_transports/default.py", line 233, in handle_request
    resp = self._pool.handle_request(req)
  File "/usr/lib/python3.8/contextlib.py", line 131, in __exit__
    self.gen.throw(type, value, traceback)
  File "/home/jesusg/.local/lib/python3.8/site-packages/httpx/_transports/default.py", line 86, in map_httpcore_exceptions
    raise mapped_exc(message) from exc
httpx.ConnectError: [Errno 111] Connection refused

* El problema viene de que no estaba instalado Ollama en el sistema, solucion para Linux:

curl -fsSL https://ollama.com/install.sh | sh

Importante, instalar el modelo que se necesite, mirar: https://github.com/ollama/ollama

Ejemplo:

ollama run llama3.1

