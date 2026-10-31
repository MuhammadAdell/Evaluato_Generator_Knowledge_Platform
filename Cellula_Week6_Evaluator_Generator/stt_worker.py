import json
import sys

from faster_whisper import WhisperModel

from config import WHISPER_MODEL


def main() -> None:
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(sys.argv[1])
    text = " ".join(segment.text.strip() for segment in segments).strip()
    print(
        json.dumps(
            {
                "text": text,
                "language": info.language,
                "duration_seconds": round(info.duration or 0, 1),
            }
        )
    )


if __name__ == "__main__":
    main()