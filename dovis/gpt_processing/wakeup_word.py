import os
import sys
import select
import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
from ament_index_python.packages import get_package_share_directory

package_path = get_package_share_directory("dovis")
MODEL_NAME = "hello_doovis_v3.tflite"
MODEL_PATH = os.path.join(package_path, f"resource/{MODEL_NAME}")


class WakeupWord:
    def __init__(self, buffer_size):
        openwakeword.utils.download_models()
        self.model = None
        self.model_name = MODEL_NAME.split(".", maxsplit=1)[0]
        self.stream = None
        self.buffer_size = buffer_size

        # 터미널에서 입력 감지를 위해 non-blocking 설정
        import tty, termios
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def is_wakeup(self):
        # 엔터 키 입력 확인
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == '\n':
                print("Enter key detected as wakeword!")
                return True

        # 오디오 스트림 분석
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        print("confidence: ", confidence)

        if confidence > 0.2:
            print("Wakeword detected!")
            return True
        return False

    def set_stream(self, stream):
        self.model = Model(wakeword_models=[MODEL_PATH])
        self.stream = stream
