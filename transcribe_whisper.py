import os
import argparse
import subprocess
from pathlib import Path
import torch
import whisper
import time
from rich.console import Console
from rich.progress import Progress

console = Console()


class WhisperTranscriber:
    def __init__(self, model_size: str = "small"):
        with console.status("[bold green]Loading Whisper model..."):
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = whisper.load_model(model_size, device=self.device)
        console.print(
            f"[bold green]✓[/bold green] Model loaded: {model_size} on {self.device}"
        )

    def convert_to_wav(self, input_path: str, max_duration: int = None) -> str:
        """Convert input media to WAV format optimized for M1."""
        output_path = str(Path(input_path).with_suffix(".wav"))
        with console.status("[bold yellow]Converting media to WAV format..."):
            command = [
                "ffmpeg",
                "-i",
                input_path,
                "-ar",
                "16000",  # Required sample rate for Whisper
                "-ac",
                "1",  # Mono audio
                "-c:a",
                "pcm_s16le",
            ]

            # Add duration limit if specified
            if max_duration:
                command.extend(["-t", str(max_duration)])

            command.extend(
                [
                    output_path,
                    "-y",  # Overwrite output file if it exists
                    "-loglevel",
                    "error",  # Reduce FFmpeg output
                ]
            )

            subprocess.run(command, check=True, capture_output=True)
        console.print("[bold green]✓[/bold green] Conversion complete")
        return output_path

    def transcribe(self, audio_path: str, language: str) -> dict:
        """Transcribe audio file."""
        console.print("\n[bold]Starting transcription...[/bold]")
        start_time = time.time()

        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Transcribing...", total=None)

            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,  # Better compatibility with MPS
                verbose=True,
            )

        elapsed_time = time.time() - start_time
        console.print(
            f"[bold green]✓[/bold green] Transcription completed in {elapsed_time:.1f} seconds"
        )
        return result

    def save_transcript(self, result: dict, output_path: str, format: str = "txt"):
        """Save transcription in specified format."""
        output_path = Path(output_path).with_suffix(f".{format}")

        with console.status(f"[bold yellow]Saving {format.upper()} transcript..."):
            if format == "txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    for segment in result["segments"]:
                        f.write(
                            f"[{segment['start']:.2f} --> {segment['end']:.2f}] {segment['text'].strip()}\n"
                        )
            elif format == "srt":
                with open(output_path, "w", encoding="utf-8") as f:
                    for i, segment in enumerate(result["segments"], 1):
                        f.write(f"{i}\n")
                        start = self._format_timestamp(segment["start"])
                        end = self._format_timestamp(segment["end"])
                        f.write(f"{start} --> {end}\n")
                        f.write(f"{segment['text'].strip()}\n\n")

        console.print(f"[bold green]✓[/bold green] Saved transcript to: {output_path}")

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio files using Whisper"
    )
    parser.add_argument("input_path", help="Path to input video/audio file")
    parser.add_argument(
        "--language", choices=["en", "es"], default="en", help="Audio language (en/es)"
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size",
    )
    parser.add_argument(
        "--format", choices=["txt", "srt"], default="txt", help="Output format"
    )
    parser.add_argument(
        "--max-duration", type=int, help="Maximum duration to process (in seconds)"
    )
    args = parser.parse_args()

    try:
        console.print(f"\n[bold]🎯 Transcribing: {Path(args.input_path).name}[/bold]\n")

        transcriber = WhisperTranscriber(model_size=args.model)
        wav_path = transcriber.convert_to_wav(args.input_path, args.max_duration)
        result = transcriber.transcribe(wav_path, args.language)

        output_path = str(Path(args.input_path).with_suffix(""))
        transcriber.save_transcript(result, output_path, args.format)

        # Cleanup
        if wav_path != args.input_path:
            os.remove(wav_path)

        console.print(
            "\n[bold green]🎉 Transcription process completed![/bold green]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise


if __name__ == "__main__":
    main()
