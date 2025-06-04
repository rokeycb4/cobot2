
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
from gpt_processing.MicController import MicController, MicConfig
from std_srvs.srv import Trigger
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from gpt_processing.wakeup_word import WakeupWord
import re
import time
from rclpy.callback_groups import ReentrantCallbackGroup
current_dir = os.getcwd()
package_path = get_package_share_directory("dovis")
is_laod = load_dotenv(dotenv_path=os.path.join(f"{package_path}/resource/.env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
class GPT(Node):
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.duration = 7  # seconds
        self.samplerate = 16000  # Whisper는 16kHz를 선호
        super().__init__("GPT")
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.5, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 협동 로봇의 조작을 돕는 dovis이다.
            <목표>
            - 문장에서 다음 리스트에 포함된 도구가 있다면 도구에 대한 대답만 해라.
            - 문장에 등장하는 도구의 목적지(어디로 옮기라고 했는지)도 함께 추출하세요.
            - 문장에서 유추할수 있는 도구가 있다면 도구에 대한 대답해라
            - 문장에서 다음 리스트에 포함된 도구가 없다면 [소통 / () ] 로 출력해라. ()안에 사용자에게 전달할 말을 넣어라.

            <도구 리스트>
            - Hammer (해머), Screwdriver (드라이버), Wrench (랜치)

            <위치 리스트>
            - 가져가 , 가져와 , pos1

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
        self.get_logger().info("wait for client's request...")
        self.get_keyword_srv = self.create_service(
            Trigger, "get_keyword", self.speech2text
        )

        self.subscription = self.create_subscription(
            String,
            'gpt_command',
            self.gpt_callback,
            10
        )

        self.personSubscription = self.create_subscription(
            Int32,
            'person_count',
            self.person_callback,
            10
        )

        self.face_publisher = self.create_publisher(Bool, 'face_command', 10)
        Mic = MicController()
        Mic.open_stream()
        self.wakeup = WakeupWord(Mic.config.buffer_size)
        self.wakeup.set_stream(Mic.stream)
        self.wake_state = True
        self.timer = self.create_timer(1.0, self.wait_pause, callback_group=ReentrantCallbackGroup())
        self.last_command_time = time.time() #  작업 공간 사람 탐지 시간
        self.warning_count = 0
        self.move_state = False
        self.motion_paused = False

    def gpt_callback(self, msg):
        command = msg.data
        self.get_logger().info(f"수신된 명령: {command}")
        self.speak(command, lang='ko')

        #사람 인식 callback
    def person_callback(self,msg):
        now = time.time()
        personCount = msg.data
        if personCount > 1:
            if self.motion_paused != True:
                self.motion_paused = True
                self.last_command_time = now
                if now - self.last_command_time > 1.0:
                    self.last_command_time = now
        else:
            if self.motion_paused != False:
                self.motion_paused = False
                self.last_command_time = now
    
    def wait_pause(self):
        if self.motion_paused:
            self.warning_count +=1 
            if self.warning_count > 5:
                self.get_logger().info("⛔ 현재 위험상태")
                self.speak('경고 현재 작업중입니다')
                self.warning_count = 0
        else:
            self.get_logger().info("✅ 안전 상태, 작업 계속")

    def speech2text(self,request, response):
        while self.wake_state:
            msg = Bool()
            msg.data = True
            self.face_publisher.publish(msg)
            if self.wakeup.is_wakeup():
                self.wake_state = False
                break
        self.move_state = True
        self.speak('무엇을 도와드릴까요?', lang='ko')
        result , transcript = self.start_speak()
        if '꺼' in transcript['text']:
            self.wake_state = True
            response.success = False
            response.message = '대기모드'
            self.move_state = False
            return response
        response.success = True
        response.message = result
        self.move_state = False
        return response
        
    def gpt_response(self, user_input):
        response = self.lang_chain.invoke({"user_input": user_input})
        result = response["text"]
        self.get_logger().info(f"GPT 응답: {result}")
        return result

    def speak(self,text, lang='ko'):
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        playsound("output.mp3")
    
    def start_speak(self):
        self.get_logger().info("음성 녹음을 시작합니다. \n 5초 동안 말해주세요...")
        audio = sd.rec(int(self.duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='int16')
        sd.wait()
        self.get_logger().info("녹음 완료. Whisper에 전송 중...")

        # 임시 WAV 파일 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav.write(temp_wav.name, self.samplerate, audio)

            # Whisper API 호출
            with open(temp_wav.name, "rb") as f:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=f,
                    api_key=self.openai_api_key
                )
        self.get_logger().info(f"STT 결과: {transcript['text']}")
        result = self.gpt_response(transcript['text'])
        return result , transcript
    
def main():
    rclpy.init()
    node = GPT(openai_api_key)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()