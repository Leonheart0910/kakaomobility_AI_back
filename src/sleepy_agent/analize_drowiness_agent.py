from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from sleepy_agent.state_agent import AgentState, LinguisticFatigueAnalysis, DrowinessScore, RoutingDecision, Quiz
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=API_KEY,
    temperature=0,
    convert_system_message_to_human=True,
)

linguistic_fatigue_analysis_model = model.with_structured_output(LinguisticFatigueAnalysis)

def analyze_user_agent(state: AgentState) -> AgentState:

    user_text = state.get("messages", [])[-1].content
    print("user_text", user_text)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 운전자의 발화를 듣고 피로도를 분석하는 전문가입니다. "
         "주어진 텍스트에서 관찰되는 언어적 특징과, 그것이 왜 피로의 신호인지 근거를 들어 분석하고, "
         "제공된 LinguisticFatigueAnalysis 형식으로 출력하세요."),
        ("human", f"이 운전자의 발화를 분석해 주세요: '{user_text}'")
    ])

    chain = prompt | linguistic_fatigue_analysis_model
    result = chain.invoke({"user_text":user_text})

    return {"linguistic_fatigue_analysis": result.model_dump()}


def route_after_analysis(state: AgentState):
    """
    졸음 분석 노드 실행 후, 점수에 따라 다음 노드를 결정하는 라우터 함수.
    """
    linguistic_fatigue_analysis = state.get("linguistic_fatigue_analysis", {})
    print("ll ", linguistic_fatigue_analysis)
    drowiness_score_model = model.with_structured_output(DrowinessScore)

    routing_model = model.with_structured_output(RoutingDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 운전자의 피로도를 분석하여 다음에 어떤 조치를 취해야 할지 결정하는 전문가입니다. "
         "주어진 언어적 특징 분석 결과를 바탕으로, 다음에 실행할 노드를 'safe_navigation_node', 'cognitive_intervention_node', 'conversation_node' 중에서 하나 선택하세요."),
        ("human", "이 분석 결과를 보고 다음에 실행할 노드를 결정해 주세요: {analysis_result}")
    ])

    chain = prompt | routing_model
    result = chain.invoke({"analysis_result": linguistic_fatigue_analysis})

    # LLM이 결정한 노드의 이름을 바로 반환
    decision = result
    print(f"--- LLM의 라우팅 결정: {decision.next_node} (근거: {decision.rationale}) ---")
    
    return decision.next_node


def summarize_messages_if_needed(state: AgentState) -> AgentState:
    """메시지 개수가 임계점을 넘으면 가장 오래된 메시지들을 요약합니다."""
    
    messages = state.get("messages", [])
    SUMMARY_THRESHOLD = 10 # 10개 메시지가 넘으면 요약 실행

    if len(messages) > SUMMARY_THRESHOLD:
        print("--- 대화 기록 요약 실행 ---")
        # 요약할 메시지들 (가장 오래된 5개)
        messages_to_summarize = messages[1:-5] # 시스템 메시지는 제외
        
        # 요약 프롬프트
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "다음 대화 내용을 한 문단으로 간결하게 요약해줘."),
            ("human", "\n".join(msg.content for msg in messages_to_summarize))
        ])
        
        # 요약 실행
        summary = (summary_prompt | model).invoke({}).content
        
        # 기존 메시지를 요약본으로 교체
        new_messages = [messages[0]] # 기존 시스템 메시지
        new_messages.append(SystemMessage(content=f"이전 대화 요약: {summary}"))
        new_messages.extend(messages[-5:]) # 최근 5개 메시지
        
        state["messages"] = new_messages
    
    return state

# Tool 정의 (실제로는 API를 호출해야 하지만, 여기서는 예시로 하드코딩)
def find_nearest_rest_area() -> str:
    """현재 위치에서 가장 가까운 졸음 쉼터 정보를 반환합니다."""
    # 실제 구현: 카카오내비 API 등을 사용하여 "OO 졸음쉼터 (X km 앞)" 형태의 정보 반환
    print("--- 졸음 쉼터 검색 툴 실행 ---")
    return "가덕해양파크 졸음쉼터 (3km 앞)"

def safe_navigation_node(state: AgentState) -> AgentState:
    print("--- 안전 운행 노드 실행 ---")
    state = summarize_messages_if_needed(state)
    
    # 1. 툴을 사용하여 가장 가까운 졸음 쉼터 검색
    rest_area_info = find_nearest_rest_area()
    
    # 2. 사용자에게 전달할 메시지 생성
    message_text = (
        "운전자님의 안전을 위해 즉시 휴식이 필요합니다. "
        f"가장 가까운 휴식 장소는 **{rest_area_info}**입니다. "
        "그 장소에서 쉬었다 가시는 건 어때요?"
    )
    
    # 3. 생성된 메시지를 messages에 추가
    return {"messages": [AIMessage(content=message_text)]}


def conversation_node(state: AgentState) -> AgentState:
    print("--- 일상 대화 노드 실행 ---")
    
    # 1. 대화 기록이 길면 요약
    state = summarize_messages_if_needed(state)
    messages = state.get("messages", [])

    # 2. LLM을 호출하여 다음 응답 생성
    response = model.invoke(messages)
    
    # 3. 생성된 응답을 messages에 추가하여 반환
    return {"messages": [response]}


def cognitive_intervention_node(state: AgentState) -> AgentState:
    print("--- 주의 주기(퀴즈) 노드 실행 ---")
    state = summarize_messages_if_needed(state)
    messages = state.get("messages")
    # 퀴즈 생성을 위한 Pydantic 모델 및 모델 체인
    quiz_model = model.with_structured_output(Quiz)
    quiz_generation_prompt = ChatPromptTemplate.from_template(
        "운전자의 정신을 환기시킬 수 있는 간단한 운전자의 기억 기반 퀴즈를 하나만 내주세요. "
        "[운전자의 기억]"
        "{messages}"
        
        "반드시 질문과 정답 목록을 포함해야 합니다. "
    )
    quiz_chain = quiz_generation_prompt | quiz_model
    # 1. 퀴즈 생성 툴 실행
    quiz_data = quiz_chain.invoke({"messages": messages}) # Quiz 객체 반환
    
    print(f"생성된 퀴즈: {quiz_data.question} (정답: {quiz_data.valid_answers})")
    
    # 2. 상태 업데이트:
    #   - messages: 사용자에게 질문을 던짐
    #   - quiz_context: 정답을 다음 턴의 검증을 위해 저장
    return {
        "messages": [AIMessage(content=quiz_data.question)],
        "quiz_context": quiz_data
    }

workflow = StateGraph(AgentState)

# 워크플로우에 각 단계(노드)를 추가합니다.
workflow.add_node("analyze_user_agent", analyze_user_agent)
workflow.add_node("route_after_analysis", route_after_analysis)
workflow.add_node("safe_navigation_node", safe_navigation_node)
workflow.add_node("cognitive_intervention_node", cognitive_intervention_node)
workflow.add_node("conversation_node", conversation_node)

workflow.add_edge(START, "analyze_user_agent")
# workflow.add_edge("analyze_user_agent", "route_after_analysis")
workflow.add_conditional_edges(
    "analyze_user_agent",
    route_after_analysis,
    {
        "safe_navigation_node": "safe_navigation_node",
        "cognitive_intervention_node": "cognitive_intervention_node",
        "conversation_node": "conversation_node",
    }
)

# workflow.add_edge("route_after_analysis", END)
workflow.add_edge("safe_navigation_node", END)
workflow.add_edge("cognitive_intervention_node", END)
workflow.add_edge("conversation_node", END)

# 워크플로우를 컴파일하여 실행 가능한 객체로 만듭니다.
law_research_agent = workflow.compile()


if __name__ == "__main__":
    law_research_agent.invoke({"messages": [HumanMessage(content="네... 그냥... 운전 중... 잘 모르겠어요.")]})
