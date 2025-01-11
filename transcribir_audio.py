import os
import subprocess
import whisper
import torch
from tqdm import tqdm
from pyannote.audio import Pipeline
from pyannote.core import Segment


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"


def split_audio(input_file, segment_length):
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


def load_diarization_pipeline(device):
    """
    Carga el pipeline de diarización de oradores.
    """
    # Reemplaza 'YOUR_HUGGINGFACE_TOKEN' con tu token de Hugging Face
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@develop",
        use_auth_token="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",
    )
    pipeline.to(device)
    return pipeline


def transcribe_and_diarize(segment_files, model, device, diarization_pipeline):
    """
    Transcribe y realiza diarización en cada segmento, asignando etiquetas de oradores.
    """
    all_segments = []
    accumulated_time = 0  # Para ajustar las marcas de tiempo

    for idx, segment_file in enumerate(
        tqdm(segment_files, desc="Procesando segmentos")
    ):
        # Diarización del segmento
        diarization = diarization_pipeline({"audio": segment_file})

        # Transcripción del segmento
        result = model.transcribe(
            segment_file, language="es", condition_on_previous_text=False
        )
        whisper_segments = result["segments"]

        # Para cada segmento de Whisper, buscamos el orador correspondiente
        for whisper_segment in whisper_segments:
            whisper_start = whisper_segment["start"]
            whisper_end = whisper_segment["end"]
            whisper_text = whisper_segment["text"].strip()

            # Ajustar los tiempos acumulados
            absolute_start = whisper_start + accumulated_time
            absolute_end = whisper_end + accumulated_time

            # Encontrar el orador correspondiente
            segment_interval = Segment(whisper_start, whisper_end)
            speaker = "Desconocido"

            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                if turn & segment_interval:
                    speaker = speaker_label
                    break

            # Crear un diccionario con toda la información
            segment_info = {
                "start": absolute_start,
                "end": absolute_end,
                "text": whisper_text,
                "speaker": speaker,
            }

            all_segments.append(segment_info)

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
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            speaker = segment["speaker"]
            text = segment["text"]
            f.write(f"[{start_time} --> {end_time}] Speaker {speaker}: {text}\n")


def clean_up():
    output_dir = "audio_chunks"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)


def main():
    input_audio_file = "input_audio.wav"
    print(input_audio_file)

    segment_length = 600
    print(segment_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = whisper.load_model("medium", device=device)
    print(f"Modelo Whisper cargado en {next(model.parameters()).device}")

    diarization_pipeline = load_diarization_pipeline(device)

    print("Dividiendo el archivo de audio en segmentos...")
    segment_files = split_audio(input_audio_file, segment_length)

    print("Transcribiendo y realizando diarización en los segmentos...")
    segments = transcribe_and_diarize(
        segment_files, model, device, diarization_pipeline
    )

    print("Guardando la transcripción completa...")
    output_file = "transcripcion_con_oradores.txt"
    save_transcription(segments, output_file)

    print(f"Transcripción completa guardada en {output_file}")

    # Limpiar archivos temporales
    clean_up()
    # os.remove(input_audio_file)


if __name__ == "__main__":
    main()
