
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import openai
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import tempfile
import os
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv
import rclpy
import pyaudio
from rclpy.node import Node
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from gtts import gTTS
from playsound import playsound
from MicController import MicController, MicConfig
from std_srvs.srv import Trigger
from std_msgs.msg import String
current_dir = os.path.dirname(os.path.abspath(__file__))
resource_path = os.path.join(current_dir, "../resource")
dotenv_path = os.path.join(resource_path, ".env")

# .env 로드
is_load = load_dotenv(dotenv_path=dotenv_path)

# 키 불러오기
openai_api_key = os.getenv("OPENAI_API_KEY")

class GPT:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.duration = 3
        self.samplerate = 16000

        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.5, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 협동 로봇의 조작을 돕는 dovis이다.
            <목표>
            - 문장에서 다음 리스트에 포함된 도구가 있다면 도구에 대한 대답만 해라.
            - 문장에 등장하는 도구의 목적지(어디로 옮기라고 했는지)도 함께 추출하세요.
            - 문장에서 유추할수 있는 도구가 있다면 그에 맞는 도구를 말해라.
            - 문장에서 다음 리스트에 포함된 도구가 없다면 [소통 / () ] 로 출력해라. ()안에 사용자에게 전달할 말을 넣어라.

            <도구 리스트>
            - Hammer (해머), Screwdriver (드라이버), Wrench (랜치)

            <위치 리스트>
            - 가져가 , 가져와 , pos1 , pos2

            <출력 형식>
            - 다음 형식을 반드시 따르세요: [도구1 도구2 ... / 위치1 위치2 ...]
            - 도구와 위치는 각각 공백으로 구분
            - 도구의 이름은 반드시 영어로 출력

            <예시>
            - 입력: "Hammer를 가져와
            출력: [Hammer / 가져와 ]

            - 입력: "Hammer를 가져가"  
            출력: [Hammer / 가져가 ]

            - 입력: "Wrench 와 Screwdriver 를 가져와"  
            출력: [Wrench Screwdriver / 가져와 가져와 ]

            - 입력: "Wrench 와 Screwdriver 를 가져가"  
            출력: [Wrench Screwdriver / 가져가 가져가 ]

            - 입력: "Hammer는 가져오고 , Screwdriver 는 가져가"
            출력: [Hammer Screwdriver / 가져와 가져가]  

            - 입력: "못 박을 도구 가져와
            출력: [Hammer / 가져와 ]

            - 입력: "나사 조일 도구 가져와
            출력: [Screwdriver / 가져와]

            - 입력: "나사 풀 도구 가져와
            출력: [Screwdriver / 가져와]

            
            <사용자 입력>
            "{user_input}"                
        """  
        self.prompt_template = PromptTemplate(
            input_variables=["user_input"], template=prompt_content
        )
        self.lang_chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

        mic_config = MicConfig(
            chunk=12000,
            rate=48000,
            channels=1,
            record_seconds=5,
            fmt=pyaudio.paInt16,
            device_index=10,
            buffer_size=24000,
        )

    def gpt_response(self, user_input):
        response = self.lang_chain.invoke({"user_input": user_input})
        result = response["text"]
        print(f"GPT 응답: {result}")
        return result

    def speak(self, text, lang='ko'):
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        playsound("output.mp3")

    def start_speak(self):
        print("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='int16')
        sd.wait()
        print("녹음 완료. Whisper에 전송 중...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)
            with open(temp_wav.name, "rb") as f:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=f,
                    api_key=self.openai_api_key
                )
        print(f"STT 결과: {transcript['text']}")
        result = self.gpt_response(transcript['text'])
    
    def get_keyword(self,text):
        import re
        match = re.search(r"\[(.*?)\]", text)
        if match:
            result = match.group(1)
            object, position = result.split("/")
            objects = object.strip().split()
            positions = position.strip().split()
            print(objects)
            print(positions)
        else:
            print("대괄호 안 텍스트가 없습니다.")
            
def main():
    gpt = GPT(openai_api_key)
    gpt.start_speak()

if __name__ == "__main__":
    main()
