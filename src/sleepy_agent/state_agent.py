from typing import Annotated, Dict, Any, TypedDict, Optional, List, Literal
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

class DrowinessScore(BaseModel):
    """
    졸음 위험도 점수를 계산하기 위한 스키마입니다.
    """
    score: int = Field(
        description="졸음 위험도 점수 (0 ~ 100)"
    )
    rationale: str = Field(
        description="점수를 산출한 근거 (디버깅 및 설명 용도)"
    )

class RoutingDecision(BaseModel):
    """
    분석 결과를 바탕으로 다음에 실행할 노드를 결정하기 위한 스키마.
    """
    next_node: Literal["safe_navigation_node", "cognitive_intervention_node", "conversation_node"] = Field(
        description="분석 결과에 따라 다음에 실행해야 할 노드의 이름."
    )
    rationale: str = Field(
        description="그러한 결정을 내린 간단한 이유."
    )

class Quiz(BaseModel):
    """퀴즈의 질문과 정답을 담는 스키마"""
    question: str
    valid_answers: List[str]


class QuizValidationResult(BaseModel):
    """퀴즈 답변의 정확도를 유연하게 판단하기 위한 스키마"""
    is_correct: bool = Field(description="사용자의 답변이 정답으로 인정될 수 있는지 여부")
    feedback: str = Field(description="채점 결과에 따라 사용자에게 전달할 피드백 메시지")
    reasoning: str = Field(description="정답 또는 오답으로 판단한 이유")


class UserIntent(BaseModel):
    """사용자 발화의 의도를 파악하기 위한 스키마"""
    is_question_to_ai: bool = Field(
        description="사용자의 발화가 AI('너', '루피')에게 직접적으로 던지는 질문인지 여부."
    )
    reasoning: str = Field(description="그렇게 판단한 이유.")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    stt_metadata: Dict[str, Any]
    # 운전 맥락 정보
    # driving_context: DrivingContext

    # 언어적 특징 분석 결과
    linguistic_fatigue_analysis: Optional[LinguisticFatigueAnalysis]
    
    quiz_timestamp: Optional[str] 
    quiz_context : str

    # # 계산된 졸음 위험도 점수 (라우팅의 기준)
    # drowsiness_score: Optional[int]
    
    # # 졸음 점수를 산출한 근거 (디버깅 및 설명 용도)
    # analysis_rationale: Optional[str]

