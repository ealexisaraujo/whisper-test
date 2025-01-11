import whisper
import torch

# Configurar el dispositivo para usar CPU
device = "cpu"

# Cargar el modelo Whisper (puedes probar con "medium" o "small" para mayor velocidad)
model = whisper.load_model("medium", device=device)

# Ruta al archivo de audio (usa el archivo WAV extraído)
file_path = "ARAAgua.wav"

# Transcribir el audio
result = model.transcribe(file_path, language="es")

# Imprimir la transcripción
print(result["text"])

# Guardar la transcripción en un archivo de texto
with open("ARAAgua.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
