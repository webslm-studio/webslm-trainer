"""Regenerate the tiny synthetic audio/video fixtures (requires ffmpeg for AVI)."""

import math
from pathlib import Path
import shutil
import struct
import subprocess
import wave


HERE = Path(__file__).resolve().parent


def make_audio() -> None:
    rate = 16000
    for name, frequencies in (("rising_tones.wav", (440, 660)),
                              ("falling_tones.wav", (660, 440))):
        samples = bytearray()
        for frequency in frequencies:
            for index in range(rate // 2):
                envelope = min(1.0, index / 300, (rate // 2 - index) / 300)
                sample = int(8000 * envelope * math.sin(2 * math.pi * frequency * index / rate))
                samples += struct.pack("<h", sample)
        with wave.open(str(HERE / name), "wb") as destination:
            destination.setnchannels(1)
            destination.setsampwidth(2)
            destination.setframerate(rate)
            destination.writeframes(samples)


def ffmpeg_executable() -> str:
    if executable := shutil.which("ffmpeg"):
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install ffmpeg or imageio-ffmpeg to regenerate video fixtures.") from exc


def make_video() -> None:
    width = height = 64
    frames = 12
    for name, moving in (("moving_square.avi", True), ("still_square.avi", False)):
        data = bytearray()
        for frame in range(frames):
            center_x = 12 + frame * 3 if moving else 30
            for y in range(height):
                for x in range(width):
                    data += b"\xff\x50\x20" if abs(x - center_x) < 8 and abs(y - 32) < 8 else b"\x14\x22\x38"
        command = [ffmpeg_executable(), "-loglevel", "error", "-y", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", "64x64", "-framerate", "12",
                   "-i", "pipe:0", "-an", "-c:v", "mjpeg", "-q:v", "5", str(HERE / name)]
        result = subprocess.run(command, input=bytes(data), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    make_audio()
    make_video()
    print(f"Generated sample media in {HERE}")
