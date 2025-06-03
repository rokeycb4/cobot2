
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
current_dir = os.getcwd()
package_path = get_package_share_directory("dovis")
is_laod = load_dotenv(dotenv_path=os.path.join(f"{package_path}/resource/.env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
class GPT(Node):
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.duration = 7  # seconds
        self.samplerate = 16000  # Whisper는 16kHz를 선호
        super().__init__("gpt_node")
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.5, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 협동 로봇의 조작을 돕는 dovis이다.
            <목표>
            - 문장에서 다음 리스트에 포함된 도구가 있다면 도구에 대한 대답만 해라.
            - 문장에 등장하는 도구의 목적지(어디로 옮기라고 했는지)도 함께 추출하세요.
            - 문장에서 유추할수 있는 도구가 있다면 도구에 대한 대답해라

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

        self.face_publisher = self.create_publisher(Bool, 'face_command', 10)
        Mic = MicController()
        Mic.open_stream()
        self.wakeup = WakeupWord(Mic.config.buffer_size)
        self.wakeup.set_stream(Mic.stream)
        self.wake_state = False

    def gpt_callback(self, msg):
        command = msg.data
        self.get_logger().info(f"수신된 명령: {command}")
        self.speak(command, lang='ko')

    def speech2text(self,request, response):
        if self.wake_state:
            self.get_logger().info("sdawaddsaadsaddaadssdasad")
            while True:
                msg = Bool()
                msg.data = True
                self.face_publisher.publish(msg)
                if self.wakeup.is_wakeup():
                    self.wake_state == False
                    break
        self.speak('무엇을 도와드릴까요?', lang='ko')
        result , transcript = self.start_speak()
        if '꺼' in transcript['text']:
            self.wake_state = True
            response.success = False
            response.message = '대기모드'
            return response
        if '이거' in transcript['text']:
            pass
        # objects,positions = self.extract_keyword(result)
        # if objects is None or positions is None:
        #     response.success = False
        #     response.message = '다시 말씀해주세요.'
        #     return response
        response.success = True
        response.message = result
        return response
        # position_keywords = ['홈', 'home']
        # if any(k in transcript['text'] for k in position_keywords):
        #     self.speak("형님, 홈 포지션으로 이동할게예~", lang='ko')
        #     response.success = True
        #     response.message = "홈"
        #     return response
        # elif '이거' in transcript['text'] or '저거' in transcript['text']:
        #     self.speak("형님, 잡아드릴게예~", lang='ko')
        #     response.success = True
        #     response.message = "이거"
        #     return response
        # elif '자동' in transcript['text']:
        #     self.speak("자동모드 입니다.", lang='ko')
        #     response.success = True
        #     response.message = "자동"
        #     return response
        # elif '망치' in transcript['text']:
        #     response.success = True
        #     response.message = "망치"
        #     return response
        # elif '가져다놔' in transcript['text'] or '가져가' in transcript['text']:
        #     response.success = True
        #     response.message = "가져가"
        #     return response
        # else:
        #     self.speak(result, lang='ko')
        #     response.success = False
        #     response.message = "도구를 찾을 수 없습니다."
        #     return response

        
    def gpt_response(self, user_input):
        response = self.lang_chain.invoke({"user_input": user_input})
        result = response["text"]
        self.get_logger().info(f"GPT 응답: {result}")
        return result

    def speak(self,text, lang='ko'):
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        playsound("output.mp3")

    def extract_keyword(self, text):
        match = re.search(r"\[(.*?)\]", text)
        if match:
            result = match.group(1)
            object, position = result.split("/")
            objects = object.strip().split()
            positions = position.strip().split()
            return objects,positions
        else:
            self.speak('다시 말해주세요.')
            return None,None
    
    def contains_dance_request(text):
        keywords = ['춤춰', '춤춰줘', '춤 보여줘', '춤 부탁해', '댄스', '춤']
        return any(keyword in text for keyword in keywords)
    
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