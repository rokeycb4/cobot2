from langchain.chat_models import ChatOpenAI  # langchain 사용 시 필요
import openai
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile
import os
from dotenv import load_dotenv

# 환경 변수 불러오기
load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# 최신 openai API에 키 설정
openai.api_key = openai_api_key


class STT:
    def __init__(self):
        self.duration = 5  # seconds
        self.samplerate = 16000  # Whisper는 16kHz를 선호

    def speech2text(self):
        print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        audio = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        print("녹음 완료. Whisper에 전송 중...")

        # 임시 WAV 파일 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)

            with open(temp_wav.name, "rb") as f:
                # 최신 방식으로 변경됨
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )

        print("STT 결과: ", transcript.text)
        return transcript.text


if __name__ == "__main__":
    stt = STT()
    output_message = stt.speech2text()
