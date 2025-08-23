from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from sleepy_agent.state_agent import AgentState, LinguisticFatigueAnalysis, DrowinessScore, RoutingDecision, Quiz, QuizValidationResult
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
from utils import get_today_str
from datetime import datetime, timezone

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
         "주어진 언어적 특징 분석 결과를 바탕으로, 다음에 실행할 노드를 'safe_navigation_node', 'cognitive_intervention_node', 'conversation_node' 중에서 하나 선택하세요."
         "'safe_navigation_node'는 즉시 쉬어야 하는 상황으로 분기할 때 사용합니다. 가까운 휴게소로 안내합니다."
         "'cognitive_intervention_node'는 사용자의 피로를 잘 가늠하기 어려울 때 분기됩니다. 사용자와의 인터랙션(질문과 답변)으로 사용자의 잠을 자연스럽게 깨웁니다."
         "'conversation_node'는 사용자가 피로하지 않거나 그 정도가 미미할 때 분기됩니다. 사용자와의 인터랙션(대화)에 집중되어 있습니다. "
         ),
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

def entry_router_node(state: AgentState):
    """그래프의 진입점에서 퀴즈 상황인지 일반 대화인지 판단하여 분기합니다."""
    print("quiz_context", state.get("quiz_context"))
    # 상태에 quiz_context가 있는지 확인
    if state.get("quiz_context"):
        print("--- 상태: 퀴즈 답변 처리 ---")
        return "validate_quiz_answer_node" # 퀴즈 검증 노드로 이동
    else:
        print("--- 상태: 일반 대화 분석 ---")
        return "analyze_user_agent" # 기존 졸음 분석 노드로 이동
    
def validate_quiz_answer_node(state: AgentState) -> AgentState:
    """퀴즈 답변의 시간 초과 여부를 확인하고, LLM으로 채점한 후 피드백을 반환합니다."""
    
    quiz_context = state["quiz_context"]
    quiz_timestamp_str = state["quiz_timestamp"]

    validation_model = model.with_structured_output(QuizValidationResult)
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 운전자의 퀴즈 답변을 너그럽게 채점하는 AI입니다. "
         "사용자의 답변이 질문의 주제와 조금이라도 관련이 있다면 '정답'으로 인정해주세요. "
         "운전자의 상태와 관계없이 단답형으로도 답변할 수 있으니 단답형 대답이나 간단한 답변도 정답으로 처리해야 합니다. "
         "무성의한 답변은 오답처리가 가능합니다. "
         "사용자의 답변이 기억을 떠올리려는 최소한의 노력만 보여도 정답입니다."),
        ("human", 
         "퀴즈 질문: '{question}'\n"
         "참고용 정답 예시: {valid_answers}\n"
         "사용자 답변: '{user_answer}'\n\n"
         "이 답변을 정답으로 처리할 수 있습니까? 결과에 맞는 피드백과 이유를 함께 알려주세요.")
    ])
    validation_chain = validation_prompt | validation_model
    
    # 1. 답변 시간 지연(Timeout) 확인
    TIMEOUT_SECONDS = 180
    quiz_time = datetime.fromisoformat(quiz_timestamp_str)
    
    if (datetime.now(timezone.utc) - quiz_time).total_seconds() > TIMEOUT_SECONDS:
        print(f"!!! {TIMEOUT_SECONDS}초 이상 응답 지연, 심각한 졸음으로 판단 !!!")
        feedback_message = AIMessage(content="괜찮으신가요? 응답이 없으셔서 걱정됩니다. 가까운 곳에서 꼭 쉬어가세요.")
    else:
        # 2. LLM을 이용한 유연한 채점
        user_answer = state.get("messages", [])[-1].content
        validation_result = validation_chain.invoke({
            "question": quiz_context.question,
            "valid_answers": quiz_context.valid_answers,
            "user_answer": user_answer
        })
        feedback_message = AIMessage(content=validation_result.feedback)

    # 3. 채점 후 퀴즈 상태를 초기화하고 피드백 메시지를 상태에 추가
    return {
        "messages": [feedback_message],
        "quiz_context": None,
        "quiz_timestamp": None
    }

def cognitive_intervention_node(state: AgentState) -> AgentState:
    print("--- 주의 주기(퀴즈) 노드 실행 ---")
    state = summarize_messages_if_needed(state)
    messages = state.get("messages")
    # 퀴즈 생성을 위한 Pydantic 모델 및 모델 체인
    quiz_model = model.with_structured_output(Quiz)
    quiz_generation_prompt = ChatPromptTemplate.from_template(
        "운전자의 정신을 환기시킬 수 있는 간단한 운전자의 기억 기반 퀴즈를 하나만 내주세요. "
        "운전자가 되도록이면 길고 성실하게 답변할 수 있는 퀴즈를 내세요."
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
        "quiz_context": quiz_data,
        "quiz_timestamp" : datetime.now(timezone.utc).isoformat()
    }

workflow = StateGraph(AgentState)

# 워크플로우에 각 단계(노드)를 추가합니다.
# workflow.add_node("entry_router_node", entry_router_node)
workflow.add_node("validate_quiz_answer_node", validate_quiz_answer_node)
workflow.add_node("analyze_user_agent", analyze_user_agent)
workflow.add_node("route_after_analysis", route_after_analysis)
workflow.add_node("safe_navigation_node", safe_navigation_node)
workflow.add_node("cognitive_intervention_node", cognitive_intervention_node)
workflow.add_node("conversation_node", conversation_node)


# workflow.add_edge(START, "entry_router_node")

workflow.add_conditional_edges(
    START,
    entry_router_node,
    {
        "validate_quiz_answer_node": "validate_quiz_answer_node",
        "analyze_user_agent": "analyze_user_agent"  # 'analyze_drowsiness_node' -> 'analyze_user_agent'로 수정
    }
)

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
    # 1. 이 대화를 식별할 고유 ID를 생성합니다.
    #    (실제 앱에서는 사용자 ID나 세션 ID를 사용할 수 있습니다.)

    memory = MemorySaver()

    agent_executor = workflow.compile(checkpointer=memory)

    conversation_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": conversation_id}}

    print(f"새로운 대화를 시작합니다. (ID: {conversation_id})")
    
    # 2. 'quit'이나 'exit'을 입력할 때까지 무한 루프를 실행합니다.
    while True:
        # 3. 사용자로부터 입력을 받습니다.
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["quit", "exit"]:
            print("대화를 종료합니다.")
            break
            
        # 4. 사용자의 입력을 HumanMessage로 변환하여 에이전트를 호출합니다.
        #    checkpointer가 config의 thread_id를 보고 이전 대화 기록을 자동으로 불러옵니다.
        final_state = agent_executor.invoke(
            {"messages": [HumanMessage(content=user_input)]}, 
            config=config
        )
        
        # 5. 에이전트의 마지막 응답을 추출하여 출력합니다.
        ai_response = final_state["messages"][-1].content
        print(f"\nAI: {ai_response}")
