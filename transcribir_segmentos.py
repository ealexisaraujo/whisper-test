import os
import subprocess
import whisper
import torch
from tqdm import tqdm


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def split_audio(input_file, segment_length):
    """
    Divide el archivo de audio en segmentos de duración especificada.

    :param input_file: Ruta al archivo de audio de entrada.
    :param segment_length: Duración de cada segmento en segundos.
    :return: Lista de rutas a los archivos de segmento generados.
    """
    output_dir = "audio_chunks"
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "ffmpeg",
        "-i",
        input_file,
        "-f",
        "segment",
        "-segment_time",
        str(segment_length),
        "-c",
        "copy",
        os.path.join(output_dir, "segment_%03d.wav"),
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    segment_files = sorted(
        [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.startswith("segment_") and f.endswith(".wav")
        ]
    )

    return segment_files


def transcribe_segments(segment_files, model, device):
    """
    Transcribe cada segmento y almacena las transcripciones con marcas de tiempo.

    :param segment_files: Lista de archivos de segmento.
    :param model: Modelo Whisper cargado.
    :param device: Dispositivo a utilizar ('cpu' o 'mps').
    :return: Lista de segmentos transcritos con marcas de tiempo.
    """
    all_segments = []
    accumulated_time = 0  # Para ajustar las marcas de tiempo

    for idx, segment_file in enumerate(
        tqdm(segment_files, desc="Transcribiendo segmentos")
    ):
        result = model.transcribe(
            segment_file, language="es", condition_on_previous_text=False
        )
        segments = result["segments"]
        # Ajustar los tiempos de inicio y fin para cada segmento
        for segment in segments:
            segment["start"] += accumulated_time
            segment["end"] += accumulated_time
            all_segments.append(segment)
        # Obtener la duración del segmento actual
        audio_info = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                segment_file,
            ],
            universal_newlines=True,
        )
        segment_duration = float(audio_info.strip())
        accumulated_time += segment_duration

    return all_segments


def save_transcription(segments, output_file):
    """
    Guarda la transcripción completa en un archivo de texto con marcas de tiempo.

    :param segments: Lista de segmentos transcritos con marcas de tiempo.
    :param output_file: Ruta al archivo de salida.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"[{start_time} --> {end_time}] {text}\n")


def clean_up():
    """
    Elimina los archivos temporales generados.
    """
    output_dir = "audio_chunks"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)


def main():
    # Ruta al archivo MP4 de entrada
    input_video_file = "ARAAgua.mp4"  # Reemplaza con la ruta a tu archivo

    # Extraer el audio del video
    input_audio_file = "input_audio.wav"
    extract_audio_command = [
        "ffmpeg",
        "-i",
        input_video_file,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        input_audio_file,
    ]
    subprocess.run(
        extract_audio_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )

    # Duración de cada segmento en segundos (5 minutos = 300 segundos)
    segment_length = 300

    # Configurar el dispositivo para usar CPU
    device = "cpu"

    # Cargar el modelo Whisper (usa un modelo más pequeño si es necesario)
    model = whisper.load_model("medium", device=device)

    print("Dividiendo el archivo de audio en segmentos...")
    segment_files = split_audio(input_audio_file, segment_length)

    print("Transcribiendo segmentos...")
    segments = transcribe_segments(segment_files, model, device)

    print("Guardando la transcripción completa...")
    output_file = "transcripcion_con_tiempos.txt"
    save_transcription(segments, output_file)

    print(f"Transcripción completa guardada en {output_file}")

    # Limpiar archivos temporales
    clean_up()
    # Eliminar el archivo de audio extraído
    os.remove(input_audio_file)


if __name__ == "__main__":
    main()
