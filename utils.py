from datetime import datetime




def get_today_str() -> str:
    """Get current date in a human-readable format. / 현재 날짜를 사람이 읽기 쉬운 형태로 반환합니다."""
    return datetime.now().strftime("%a %b %-d, %Y")  


# 파일 상단에 추가
# import elevenlabs
from dotenv import load_dotenv
import uuid
import os
from elevenlabs.client import ElevenLabs # 변경점 1: 클라이언트를 import 합니다.
from elevenlabs import Voice, VoiceSettings, save 
from normalization import convert_numbers_in_string_to_korean

load_dotenv()

# ElevenLabs 클라이언트 설정
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

LUFFY_VOICE_ID = "h1nAmMChLiUounDhPXJ1" # 사용하실 Voice ID로 교체하세요.
MODEL_ID = "eleven_multilingual_v2"
VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True
}

# --- 최종 수정된 함수 ---
async def generate_luffy_audio(text: str) -> str:
    """
    텍스트를 받아 ElevenLabs API로 오디오를 생성하고 파일로 저장합니다.
    (사용자가 제공한 코드를 기반으로 재작성됨)
    """
    try:
        # 1. client.text_to_speech.convert API를 사용하여 오디오 스트림 생성
        audio_stream = client.text_to_speech.convert(
            voice_id=LUFFY_VOICE_ID,
            model_id=MODEL_ID,
            text=convert_numbers_in_string_to_korean(text),
            voice_settings=VOICE_SETTINGS,
            output_format="mp3_44100_128" # 필요 시 출력 포맷 지정
        )

        # 2. 스트리밍된 오디오 청크(chunk)들을 하나로 합쳐 bytes 데이터로 변환
        audio_bytes = b"".join(audio_stream)

        # 3. 고유한 파일 이름을 생성하고 저장 경로 설정
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join("static", filename)
        
        # 'static' 폴더가 없으면 생성
        os.makedirs("static", exist_ok=True)

        # 4. bytes 데이터를 바이너리 쓰기 모드('wb')로 파일에 저장
        with open(filepath, "wb") as f:
            f.write(audio_bytes)

        print(f"✅ 오디오 파일 저장 완료: {filepath}")
        return filename

    except Exception as e:
        print(f"❌ ElevenLabs API 오류 발생: {e}")
        return ""