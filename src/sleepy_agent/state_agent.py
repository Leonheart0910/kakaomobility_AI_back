from typing import Annotated, Dict, Any, TypedDict, Optional, List
from pydantic import BaseModel, Field
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

    # 계산된 졸음 위험도 점수 (라우팅의 기준)
    drowsiness_score: Optional[int]
    
    # 졸음 점수를 산출한 근거 (디버깅 및 설명 용도)
    analysis_rationale: Optional[str]


class LinguisticFatigueAnalysis(BaseModel):
    """
    사용자 발화의 언어적 특징을 분석하여 피로도를 측정하기 위한 스키마입니다.
    """
    
    linguistic_features: List[str] = Field(
        description="피로 상태로 판단할 수 있는 관찰된 언어적 특징들의 목록. 예: ['단답형 대답', '느린 응답 속도', '횡설수설']"
    )
    reasoning: str = Field(
        description="위 언어적 특징들이 왜 피로의 신호가 되는지에 대한 논리적인 근거."
    )