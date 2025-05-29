import time
import openai
import requests
import os
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
from dotenv import load_dotenv

import wakeup_word  # 같은 디렉토리에서 import
import MicController

# .env에서 API 키 로드
load_dotenv(dotenv_path=".env")
openai.api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")
voice_id = os.getenv("ELEVEN_VOICE_ID")


# 웨이크워드 대기
def wait_for_wakeword():
    Mic = MicController.MicController()
    Mic.open_stream()

    wakeup = wakeup_word.WakeupWord(Mic.config.buffer_size)
    wakeup.set_stream(Mic.stream)

    print("🕓 웨이크워드(예: Hello Doovis)를 기다리는 중...")
    while True:
        if wakeup.is_wakeup():
            break
        time.sleep(0.1)


# 음성 → 텍스트
def speech_to_text(duration=5, samplerate=16000):
    print("🎙️ 5초 동안 말해주세요...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    print("✅ 녹음 완료. Whisper로 전송 중...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        wav.write(temp_wav.name, samplerate, audio)
        with open(temp_wav.name, "rb") as f:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)

    print("📝 인식된 질문:", transcript.text)
    return transcript.text


# GPT 응답
def ask_gpt(prompt):
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are JARVIS, a formal and helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# 자비스 목소리로 말하기
def speak_jarvis(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": eleven_api_key,
        "Content-Type": "application/json"
    }
    body = {
        "text": text,
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.8
        }
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        with open("jarvis_voice.mp3", "wb") as f:
            f.write(response.content)
        os.system("mpg123 jarvis_voice.mp3")
    else:
        print("❌ TTS 실패:", response.status_code, response.text)


# 웨이크워드 후 첫 인사
def speak_jarvis_wakeup():
    speak_jarvis("At your service, sir.")


# 전체 흐름
if __name__ == "__main__":
    wait_for_wakeword()
    speak_jarvis_wakeup()

    while True:
        input("⏎ Enter를 눌러 질문을 시작하세요...")
        user_input = speech_to_text()
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("🛑 Jarvis 종료")
            break
        gpt_response = ask_gpt(user_input)
        print("🤖 Jarvis:", gpt_response)
        speak_jarvis(gpt_response)
