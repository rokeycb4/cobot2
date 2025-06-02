
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
current_dir = os.getcwd()
package_path = get_package_share_directory("dovis")
is_laod = load_dotenv(dotenv_path=os.path.join(f"{package_path}/resource/.env"))
openai_api_key = os.getenv("OPENAI_API_KEY")
class GPT(Node):
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.duration = 3  # seconds
        self.samplerate = 16000  # Whisper는 16kHz를 선호
        super().__init__("gpt_node")
        self.llm = ChatOpenAI(
            model="gpt-4o", temperature=0.5, openai_api_key=openai_api_key
        )

        prompt_content = """
            당신은 협동 로봇의 조작을 돕는 dovis이다. 부산 사투리를 사용하여라.
            <목표>
            - 문장에서 다음 리스트에 포함된 도구가 있다면 도구에 대한 대답만 해라.
            - 문장에서 다음 리스트에 포함된 도구가 없다면 사용자와 소통하여라.
            - 문장에 등장하는 도구의 목적지(어디로 옮기라고 했는지)도 함께 추출하세요.

            <도구 리스트>
            - Hammer, Screwdriver, Wrench

            <위치 리스트>
            - pos1, pos2, pos3, pos4, pos5

            <출력 형식>
            - 다음 형식을 반드시 따르세요: [도구1 도구2 ... / 위치1 위치2 ...]
            - 도구와 위치는 각각 공백으로 구분

            <예시>
            - 입력: "Hammer를 pos1에 가져다 놔"  
            출력: [Hammer / pos1]

            - 입력: "왼쪽에 있는 해머와 Wrench를 pos1에 넣어줘"  
            출력: [Hammer wrench / pos1]

            - 입력: "왼쪽에 있는 해머는 pos1에 넣고, 오른쪽에 있는 Wrench는 pos2에 넣어줘"
            출력: [Hammer Wrench / pos1 pos2]  
            
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

    def gpt_callback(self, msg):
        command = msg.data
        self.get_logger().info(f"수신된 명령: {command}")
        self.speak(command, lang='ko')

    def speech2text(self,request, response):
        while True:
            self.speak('무엇을 도와드릴까요?', lang='ko')
            confirm = input("형님, 음성 인식을 시작할까요? (y/n): ").strip().lower()
            if confirm == 'y':
                break
        # 녹음 설정
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
        import re
        match = re.search(r'\[([^\[\]]+)\]', result)
        if match:
            keyword_text = match.group(1).strip()
            keyword = self.extract_keyword(keyword_text)
            spoken_text = re.sub(r'\[[^\[\]]+\]', '', result).strip()
            response.success = True
            response.message = " ".join(keyword) 
            if spoken_text:
                self.speak(spoken_text)
            return response
        else:
            dance_keywords = ['춤', '춤춰', '춤춰줘', '춤 보여줘', '댄스']
            position_keywords = ['홈', 'home']
            if any(k in result for k in dance_keywords):
                self.speak("형님, 신나는 춤 바로 보여드릴게예~", lang='ko')
                response.success = True
                response.message = "춤"  # 👉 로봇 노드에서 이걸 기반으로 춤 동작 수행
                return response
            elif any(k in transcript['text'] for k in position_keywords):
                self.speak("형님, 홈 포지션으로 이동할게예~", lang='ko')
                response.success = True
                response.message = "홈"
                return response
            elif '이거' in transcript['text'] or '저거' in transcript['text']:
                self.speak("형님, 잡아드릴게예~", lang='ko')
                response.success = True
                response.message = "이거"
                return response
            elif '자동' in transcript['text']:
                self.speak("자동모드 입니다.", lang='ko')
                response.success = True
                response.message = "자동"
                return response
            elif '망치' in transcript['text']:
                response.success = True
                response.message = "망치"
                return response
            elif '가져다놔' in transcript['text'] or '가져가' in transcript['text']:
                response.success = True
                response.message = "가져가"
                return response
            else:
                self.speak(result, lang='ko')
                response.success = False
                response.message = "도구를 찾을 수 없습니다."
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

    def extract_keyword(self, result):
        object, target = result.strip().split("/")

        object = object.split()
        target = target.split()

        self.get_logger().info(f"llm's response: {object}")
        self.get_logger().info(f"object: {object}")
        self.get_logger().info(f"target: {target}")
        return object
    
    def contains_dance_request(text):
        keywords = ['춤춰', '춤춰줘', '춤 보여줘', '춤 부탁해', '댄스', '춤']
        return any(keyword in text for keyword in keywords)
    
def main():
    rclpy.init()
    node = GPT(openai_api_key)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()