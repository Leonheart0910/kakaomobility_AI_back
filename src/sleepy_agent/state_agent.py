from typing import Annotated, Dict, Any, TypedDict, Optional
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END


class SttMetadata(TypedDict):
    response_time_ms: int      # 사용자가 응답하기까지 걸린 시간 (밀리초)
    is_recognized: bool        # 음성 인식 성공 여부
    confidence_score: float    # 인식 결과의 신뢰도 점수 (0.0 ~ 1.0)

# 운전 맥락 정보를 담을 데이터 구조 정의
class DrivingContext(TypedDict):
    current_time_iso: str      # 현재 시간 (ISO 8601 형식)
    total_driving_minutes_today: int # 오늘 누적 운전 시간 (분)

class AgentState(StateGraph):
    messages: Annotated[list, add_messages]

    stt_metadata: Dict[str, Any]
    # 운전 맥락 정보
    # driving_context: DrivingContext

    drowsiness_score: Optional[int]

    analysis_rationale: Optional[str]


