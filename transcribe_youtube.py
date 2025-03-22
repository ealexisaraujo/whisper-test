import os, argparse, subprocess, tempfile, time, shutil
from pathlib import Path
import torch
import whisper
import yt_dlp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler
import logging

# Configure logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("whisper_transcriber")
console = Console()


class WhisperTranscriber:
    def __init__(self, model_size: str = "small"):
        logger.info("Initializing Whisper Transcriber...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        with console.status("[bold green]Loading Whisper model...") as status:
            try:
                self.model = whisper.load_model(model_size, device=self.device)
                logger.info(f"✓ Successfully loaded {model_size} model")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise

    def download_youtube(
        self,
        url: str,
        output_dir: str,
        cookies_file: str = None,
        browser_cookies: str = None,
    ) -> str:
        """Step 1: Download YouTube video audio

        Args:
            url: YouTube URL to download
            output_dir: Directory to save downloaded audio
            cookies_file: Path to cookies file for authentication
            browser_cookies: Browser name to extract cookies from (chrome, firefox, etc.)
        """
        logger.info("\n=== Step 1: YouTube Download ===")
        logger.info(f"Processing URL: {url}")
        logger.info(f"Output directory: {output_dir}")

        # Debug log for cookie options
        if cookies_file:
            logger.info(f"Using cookies file: {cookies_file}")
        if browser_cookies:
            logger.info(f"Using cookies from browser: {browser_cookies}")

        timestamp = int(time.time())
        output_template = os.path.join(output_dir, f"audio_{timestamp}.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
            "progress_hooks": [self._download_progress_hook],
            "logger": logger,
            # Adding custom HTTP headers to spoof a browser user-agent
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
            },
        }

        # Add cookies configuration if provided
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file
        if browser_cookies:
            ydl_opts["cookiesfrombrowser"] = (browser_cookies,)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Starting download process...")
                info = ydl.extract_info(url, download=True)
                title = info.get("title", "Unknown Title")
                logger.info(f"Video title: {title}")

                downloaded_file = os.path.join(output_dir, f"audio_{timestamp}.m4a")
                if not os.path.exists(downloaded_file):
                    raise FileNotFoundError(
                        f"Expected file not found: {downloaded_file}"
                    )

                file_size = os.path.getsize(downloaded_file) / (
                    1024 * 1024
                )  # Size in MB
                logger.info(
                    f"✓ Download complete: {downloaded_file} ({file_size:.1f} MB)"
                )
                return downloaded_file

        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise

    def _download_progress_hook(self, d):
        if d["status"] == "downloading":
            try:
                percent = d["_percent_str"]
                speed = d.get("_speed_str", "N/A")
                logger.info(
                    f"Download Progress: {percent} Speed: {speed}",
                    extra={"markup": True},
                )
            except:
                pass

    def convert_to_wav(self, input_path: str) -> str:
        """Step 2: Convert audio to WAV format"""
        logger.info("\n=== Step 2: Audio Conversion ===")
        logger.info(f"Input file: {input_path}")

        output_path = str(Path(input_path).with_suffix(".wav"))
        logger.info(f"Converting to: {output_path}")

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

        try:
            logger.info("Starting conversion process...")
            subprocess.run(command, check=True, capture_output=True)

            if not os.path.exists(output_path):
                raise FileNotFoundError("WAV conversion failed")

            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            logger.info(f"✓ Conversion complete: {output_path} ({file_size:.1f} MB)")
            return output_path

        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            raise

    def transcribe(self, audio_path: str, language: str) -> dict:
        """Step 3: Transcribe audio"""
        logger.info("\n=== Step 3: Transcription ===")
        logger.info(f"Processing file: {audio_path}")
        logger.info(f"Target language: {language}")

        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            start_time = time.time()
            logger.info("Starting transcription...")

            result = self.model.transcribe(
                audio_path, language=language, fp16=False, verbose=True
            )

            elapsed = time.time() - start_time
            logger.info(f"✓ Transcription completed in {elapsed:.1f} seconds")

            # Log some statistics about the transcription
            segment_count = len(result["segments"])
            total_duration = sum(
                seg["end"] - seg["start"] for seg in result["segments"]
            )
            logger.info(f"Generated {segment_count} segments")
            logger.info(f"Total audio duration: {total_duration:.1f} seconds")

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def save_transcript(self, result: dict, output_path: str, format: str = "txt"):
        """Step 4: Save transcription"""
        logger.info(f"\n=== Step 4: Saving Transcript ===")
        logger.info(f"Output format: {format.upper()}")

        output_path = Path(output_path).with_suffix(f".{format}")
        logger.info(f"Saving to: {output_path}")

        try:
            segment_count = 0
            with open(output_path, "w", encoding="utf-8") as f:
                if format == "txt":
                    for segment in result["segments"]:
                        f.write(
                            f"[{segment['start']:.2f} --> {segment['end']:.2f}] {segment['text'].strip()}\n"
                        )
                        segment_count += 1
                else:  # srt
                    for i, segment in enumerate(result["segments"], 1):
                        f.write(f"{i}\n")
                        start = self._format_timestamp(segment["start"])
                        end = self._format_timestamp(segment["end"])
                        f.write(f"{start} --> {end}\n")
                        f.write(f"{segment['text'].strip()}\n\n")
                        segment_count += 1

            file_size = os.path.getsize(output_path) / 1024  # Size in KB
            logger.info(f"✓ Saved {segment_count} segments ({file_size:.1f} KB)")

        except Exception as e:
            logger.error(f"Failed to save transcript: {str(e)}")
            raise

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio files using Whisper"
    )
    parser.add_argument("input", help="Path to input video/audio file or YouTube URL")
    parser.add_argument(
        "--language", choices=["en", "es"], default="en", help="Audio language"
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
    # Add YouTube authentication options
    parser.add_argument(
        "--cookies", help="Path to cookies file for YouTube authentication"
    )
    parser.add_argument(
        "--browser-cookies",
        choices=["chrome", "firefox", "opera", "edge", "safari"],
        help="Browser to extract cookies from for YouTube authentication",
    )
    args = parser.parse_args()

    logger.info("\n=== Whisper Transcription Pipeline Started ===")
    logger.info(f"Input: {args.input}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output format: {args.format}")

    # Log cookie options if provided
    if args.cookies:
        logger.info(f"Cookies file: {args.cookies}")
    if args.browser_cookies:
        logger.info(f"Browser cookies: {args.browser_cookies}")

    is_youtube = args.input.startswith(("http://", "https://")) and (
        "youtube.com" in args.input or "youtu.be" in args.input
    )

    if is_youtube:
        logger.info("Detected YouTube URL input")
    else:
        logger.info("Detected local file input")
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return

    temp_dir = None
    audio_file = None
    wav_file = None

    try:
        transcriber = WhisperTranscriber(model_size=args.model)

        if is_youtube:
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory: {temp_dir}")

            # Pass cookie information to download_youtube
            audio_file = transcriber.download_youtube(
                args.input,
                temp_dir,
                cookies_file=args.cookies,
                browser_cookies=args.browser_cookies,
            )
            wav_file = transcriber.convert_to_wav(audio_file)
            result = transcriber.transcribe(wav_file, args.language)
            output_base = os.path.join(
                os.getcwd(), f"youtube_transcript_{int(time.time())}"
            )
        else:
            result = transcriber.transcribe(args.input, args.language)
            output_base = str(Path(args.input).with_suffix(""))

        transcriber.save_transcript(result, output_base, args.format)
        logger.info("\n=== Pipeline Completed Successfully ===")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

    finally:
        logger.info("\n=== Cleaning Up ===")
        files_cleaned = 0
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
            files_cleaned += 1
        if wav_file and os.path.exists(wav_file):
            os.remove(wav_file)
            files_cleaned += 1
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up {files_cleaned} temporary files")


if __name__ == "__main__":
    main()
