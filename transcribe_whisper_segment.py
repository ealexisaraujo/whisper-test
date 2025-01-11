import os
import argparse
import subprocess
from pathlib import Path
import torch
import whisper
import time
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from concurrent.futures import ThreadPoolExecutor
import tempfile

console = Console()


class WhisperTranscriber:
    def __init__(self, model_size: str = "small"):
        with console.status("[bold green]Loading Whisper model..."):
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = whisper.load_model(model_size, device=self.device)
        console.print(
            f"[bold green]✓[/bold green] Model loaded: {model_size} on {self.device}"
        )

    def split_video(self, input_path: str, segment_duration: int = 600) -> list:
        """Split video into segments of specified duration."""
        temp_dir = tempfile.mkdtemp()
        console.print(
            f"[bold yellow]Splitting video into {segment_duration} second segments...[/bold yellow]"
        )

        # Get video duration
        duration_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_path,
        ]
        duration = float(subprocess.check_output(duration_cmd).decode("utf-8").strip())

        segment_pattern = os.path.join(temp_dir, "segment_%03d.mp4")
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-f",
            "segment",
            "-segment_time",
            str(segment_duration),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            segment_pattern,
        ]

        subprocess.run(command, check=True, capture_output=True)

        # Get list of generated segments
        segments = sorted(
            [
                os.path.join(temp_dir, f)
                for f in os.listdir(temp_dir)
                if f.startswith("segment_") and f.endswith(".mp4")
            ]
        )

        console.print(f"[bold green]✓[/bold green] Split into {len(segments)} segments")
        return segments, temp_dir

    def convert_to_wav(self, input_path: str) -> str:
        """Convert input media to WAV format optimized for M1."""
        output_path = str(Path(input_path).with_suffix(".wav"))
        with console.status(
            f"[bold yellow]Converting {Path(input_path).name} to WAV..."
        ):
            command = [
                "ffmpeg",
                "-i",
                input_path,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                output_path,
                "-y",
                "-loglevel",
                "error",
            ]
            subprocess.run(command, check=True, capture_output=True)
        return output_path

    def process_segment(self, segment_file: str, language: str) -> dict:
        """Process a single segment."""
        wav_path = self.convert_to_wav(segment_file)
        result = self.model.transcribe(
            wav_path, language=language, fp16=False, verbose=False
        )
        os.remove(wav_path)
        return result

    def transcribe_segmented(
        self, input_path: str, language: str, segment_duration: int = 600
    ) -> dict:
        """Transcribe using segmented approach."""
        start_time = time.time()

        # Split video into segments
        segments, temp_dir = self.split_video(input_path, segment_duration)

        all_segments = []
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Transcribing segments...", total=len(segments)
            )

            # Process each segment
            for segment in segments:
                result = self.process_segment(segment, language)
                all_segments.extend(result["segments"])
                progress.advance(task)

        # Cleanup temporary files
        shutil.rmtree(temp_dir)

        elapsed_time = time.time() - start_time
        console.print(
            f"[bold green]✓[/bold green] Transcription completed in {elapsed_time:.1f} seconds"
        )

        return {"segments": all_segments}

    def transcribe(self, audio_path: str, language: str) -> dict:
        """Traditional single-file transcription."""
        console.print("\n[bold]Starting transcription...[/bold]")
        start_time = time.time()

        wav_path = self.convert_to_wav(audio_path)
        with Progress(console=console) as progress:
            progress.add_task("[cyan]Transcribing...", total=None)
            result = self.model.transcribe(
                wav_path, language=language, fp16=False, verbose=True
            )

        # Cleanup
        os.remove(wav_path)

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
        description="""
Whisper Audio/Video Transcription Tool

This script transcribes audio/video files using OpenAI's Whisper model.

Example usage:
    python transcribe_whisper_segment.py input.mp4 --language en --model base --format txt
    python transcribe_whisper_segment.py podcast.mp3 --language es --model small --segment
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        help="Path to the input video/audio file (supports mp4, mp3, wav, etc.)",
    )
    parser.add_argument(
        "--language",
        choices=["en", "es"],
        default="en",
        help="Language of the audio content (en=English, es=Spanish)",
    )
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size. Larger models are more accurate but slower",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "srt"],
        default="txt",
        help="Output format: txt (timestamped text) or srt (subtitle format)",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Enable segmented processing for long videos (recommended for files > 10 minutes)",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=600,
        help="Duration of each segment in seconds when using --segment (default: 600)",
    )
    args = parser.parse_args()

    try:
        console.print(f"\n[bold]🎯 Transcribing: {Path(args.input_path).name}[/bold]\n")

        transcriber = WhisperTranscriber(model_size=args.model)

        if args.segment:
            result = transcriber.transcribe_segmented(
                args.input_path, args.language, args.segment_duration
            )
        else:
            result = transcriber.transcribe(args.input_path, args.language)

        output_path = str(Path(args.input_path).with_suffix(""))
        transcriber.save_transcript(result, output_path, args.format)

        console.print(
            "\n[bold green]🎉 Transcription process completed![/bold green]\n"
        )

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}\n")
        raise


if __name__ == "__main__":
    main()
