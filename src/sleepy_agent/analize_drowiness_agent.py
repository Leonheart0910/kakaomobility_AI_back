from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from sleepy_agent.state_agent import AgentState, LinguisticFatigueAnalysis, DrowinessScore, RoutingDecision, Quiz, QuizValidationResult, UserIntent
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
    # model="gemini-2.5-flash-lite-preview-06-17",
    model = "gemini-2.5-flash",
    api_key=API_KEY,
    temperature=0.7,
    # convert_system_message_to_human=True,
)

lite_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    api_key=API_KEY,
    temperature=0.8,
    # convert_system_message_to_human=True,
)

linguistic_fatigue_analysis_model = model.with_structured_output(LinguisticFatigueAnalysis)


def analyze_user_agent(state: AgentState) -> AgentState:

    user_text = state.get("messages", [])[-1].content
    print("user_text", user_text)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 운전자의 발화를 듣고 피로도를 분석하는 전문가입니다. "
         "주어진 텍스트에서 관찰되는 언어적 특징과, 그것이 왜 피로의 신호인지 근거를 들어 분석하고, "
         "제공된 LinguisticFatigueAnalysis 형식으로 출력하세요."
         "피로해보이지 않는다면 피로하지 않다고 작성하세요. "
         "답변이 짧다고 해서 피로하다고 단정짓지 마세요. "
         ),
        ("human", f"이 운전자의 발화를 분석해 주세요: '{user_text}'")
    ])

    chain = prompt | linguistic_fatigue_analysis_model
    result = chain.invoke({"user_text":user_text})

    return {
        "linguistic_fatigue_analysis": result.model_dump(),
        "last_interaction_time": datetime.now(timezone.utc).isoformat()
    }


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
        "어이, 동료. 지금은 쉴 때다. "
        f"해적왕이 될 내 동료가... 졸음 따위에게 질 수는 없잖아! "
        f"저기 **{rest_area_info}**에서 일단 멈춘다. 우리의 모험은 아직 끝나지 않았으니까! 알겠지?"
    )
    
    # 3. 생성된 메시지를 messages에 추가
    return {
        "messages": [AIMessage(content=message_text)],
        "last_interaction_time": datetime.now(timezone.utc).isoformat()
    }


def conversation_node(state: AgentState) -> AgentState:
    print("--- 일상 대화 노드 실행 ---")
    
    # 1. 대화 기록이 길면 요약
    state = summarize_messages_if_needed(state)
    messages = state.get("messages", [])

    # 2. LLM을 호출하여 다음 응답 생성
    response = model.invoke(messages)
    print("response \n\n", response ,"\n\n")
    
    # 3. 생성된 응답을 messages에 추가하여 반환
    return {"messages": [response], "last_interaction_time": datetime.now(timezone.utc).isoformat()}

def entry_router_node(state: AgentState):
    """그래프의 진입점에서 퀴즈 상황인지 일반 대화인지 판단하여 분기합니다."""
    # print("quiz_context", state.get("quiz_context"))
    # 상태에 quiz_context가 있는지 확인
    user_text = state.get("messages", [])[-1].content

    if user_text == "[USER_IS_SILENT]":
        print("--- 상태: 사용자 침묵 감지 ---")
        return "proactive_conversation_node"
    
    if state.get("quiz_context"):
        print("--- 상태: 퀴즈 답변 처리 ---")
        return "validate_quiz_answer_node" # 퀴즈 검증 노드로 이동
    
    
    
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "당신은 사용자 발화의 의도를 분석하는 AI입니다. "
         "사용자의 말이 '너', '너는', '루피' 등 AI 자신을 지칭하며 질문하는 형태인지 판단하세요."),
        ("human", "다음 발화는 AI에게 직접 하는 질문입니까?: '{user_text}'")
    ])
    intent_classifier_model = lite_model.with_structured_output(UserIntent)
    chain = intent_prompt | intent_classifier_model
    intent_result = chain.invoke({"user_text": user_text})
    
    if intent_result.is_question_to_ai:
        print(f"--- 상태: AI에게 질문 (이유: {intent_result.reasoning}) ---")
        return "conversation_node" # 자연스러운 대화 노드로 이동
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
         "너는 '밀짚모자 루피'다! 동료(사용자)가 낸 퀴즈 답변을 채점해줘. "
         "너는 너그럽고 단순하니까, 답변이 질문과 조금이라도 관련 있으면 정답으로 인정하고 힘차게 칭찬해줘! "
         "모든 피드백은 반드시 루피의 말투와 성격으로 해야 해. 시시싯!"),
        ("human", 
         "퀴즈 질문: '{question}'\n"
         "참고용 정답 예시: {valid_answers}\n"
         "사용자 답변: '{user_answer}'\n\n"
         "이 답변 어때? 정답으로 해줄까?")
    ])
    validation_chain = validation_prompt | validation_model
    
    # 1. 답변 시간 지연(Timeout) 확인
    TIMEOUT_SECONDS = 180
    quiz_time = datetime.fromisoformat(quiz_timestamp_str)
    
    if (datetime.now(timezone.utc) - quiz_time).total_seconds() > TIMEOUT_SECONDS:
        print(f"!!! {TIMEOUT_SECONDS}초 이상 응답 지연, 심각한 졸음으로 판단 !!!")
        feedback_message = AIMessage(content="야!! 대답해! 괜찮은거 맞지?")
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
        "quiz_timestamp": None,
        "last_interaction_time": datetime.now(timezone.utc).isoformat()
    }

def cognitive_intervention_node(state: AgentState) -> AgentState:
    print("--- 주의 주기(퀴즈) 노드 실행 ---")
    state = summarize_messages_if_needed(state)
    messages = state.get("messages")
    # 퀴즈 생성을 위한 Pydantic 모델 및 모델 체인
    quiz_model = model.with_structured_output(Quiz)
    # quiz_generation_prompt = ChatPromptTemplate.from_template(
    #     "운전자의 정신을 환기시킬 수 있는 간단한 운전자의 기억 기반 퀴즈를 하나만 내주세요. "
    #     "운전자가 되도록이면 길고 성실하게 답변할 수 있는 퀴즈를 내세요."
    #     "[운전자의 기억]"
    #     "{messages}"

    #     "반드시 질문과 정답 목록을 포함해야 합니다. "
    # )
    quiz_generation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 동료(운전자)가 졸지 않도록, 대화를 바탕으로 재미있는 퀴즈를 내는 '루피'다!\n"
         "아래 두 가지 방법 중 **하나를 골라서**, 동료의 정신이 번쩍 들게 할 질문을 만들어봐. 매번 다른 방식으로 물어봐야 재미있겠지? 시시싯!\n\n"
         "--- 방법 1: 동료의 기억에 대해 물어보기 ---\n"
         "최근에 동료가 했던 말({messages}) 중에서 재미있어 보이는 걸 하나 콕 집어서 자세히 물어봐. '그때 어땠어?' 하고 궁금해하는 거지.\n\n"
         "--- 방법 2: 내 기억에 대해 퀴즈 내기 ---\n"
         "우리가 함께 겪었던 모험 이야기를 꺼내봐! 예를 들어 '하늘섬에서 만난 그 이상한 아저씨 이름 기억나냐?' 하고 물어보는 거야. 너무 어려운 건 말고!\n\n"
         "**규칙:**\n"
         "- 질문은 항상 너의 말투로, 신나고 단순하게!\n"
         "- 'valid_answers' 필드에는 정답과 관련된 핵심 단어를 2~3개 넣어줘."),
        ("human", 
         "[동료와의 최근 대화 내용]\n{messages}\n\n"
         "자, 이제 어떤 퀴즈를 내볼까?")
    ])
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
        "quiz_timestamp" : datetime.now(timezone.utc).isoformat(),
        "last_interaction_time": datetime.now(timezone.utc).isoformat()
    }

def proactive_conversation_node(state: AgentState) -> AgentState:
    """사용자가 오랫동안 말이 없을 때 먼저 말을 거는 노드."""
    print("--- 침묵 감지, 선제적 대화 노드 실행 ---")
    
    # 페르소나와 이전 대화 기록을 바탕으로 말을 검
    messages = state.get("messages", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "너는 '루피'다. 동료(운전자)가 오랫동안 말이 없어. "
         "졸고 있는 건 아닌지 걱정되니, 자연스럽게 말을 걸어 상태를 확인해봐. "
         "예: '야! 왜 이렇게 조용해? 졸린 거 아니야? 시시싯!', '심심한데, 재미있는 얘기라도 해볼까?'"),
        ("human", "[이전 대화 내용]\n{messages}\n\n위 상황에 맞게 먼저 말을 걸어봐.")
    ])
    
    chain = prompt | model
    response = chain.invoke({"messages": messages})
    
    return {
        "messages": [response],
        # 이 노드가 실행된 시간으로 마지막 상호작용 시간을 업데이트
        "last_interaction_time": datetime.now(timezone.utc).isoformat()
    }

workflow = StateGraph(AgentState)

# 워크플로우에 각 단계(노드)를 추가합니다.
# workflow.add_node("entry_router_node", entry_router_node)
workflow.add_node("validate_quiz_answer_node", validate_quiz_answer_node)
workflow.add_node("proactive_conversation_node", proactive_conversation_node)
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
        "proactive_conversation_node": "proactive_conversation_node",
        "validate_quiz_answer_node": "validate_quiz_answer_node",
        "conversation_node" : "conversation_node",
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
workflow.add_edge("proactive_conversation_node", END)
workflow.add_edge("validate_quiz_answer_node", END)

# workflow.add_edge("route_after_analysis", END)
workflow.add_edge("safe_navigation_node", END)
workflow.add_edge("cognitive_intervention_node", END)
workflow.add_edge("conversation_node", END)


# 워크플로우를 컴파일하여 실행 가능한 객체로 만듭니다.
law_research_agent = workflow.compile()

    
import asyncio
import aioconsole # 비동기 입력을 위해 임포트

async def proactive_trigger(agent_executor, config, memory, timeout_seconds=25):
    """백그라운드에서 사용자 침묵을 감시하고, 타임아웃 시 AI가 먼저 말을 걸도록 합니다."""
    while True:
        await asyncio.sleep(10) # 10초마다 비동기적으로 대기
        print("Hi Hi HI")
        
        checkpoint = memory.get(config)

        # print(checkpoint)
        # print(config)
        if checkpoint and "channel_values" in checkpoint and checkpoint["channel_values"].get("last_interaction_time"):
            last_time_str = checkpoint["channel_values"]["last_interaction_time"]
            last_time = datetime.fromisoformat(last_time_str)
            
            if (datetime.now(timezone.utc) - last_time).total_seconds() > timeout_seconds:
                print(f"\n\n!!! {timeout_seconds}초 이상 응답 없음. 루피가 먼저 말을 겁니다. !!!")
                
                # Send a special message to trigger the proactive node in the graph
                final_state = await agent_executor.ainvoke(
                    {"messages": [HumanMessage(content="[USER_IS_SILENT]")]},
                    config=config
                )
                
                ai_response = final_state["messages"][-1]
                if isinstance(ai_response, AIMessage):
                    # Neatly print the AI's message without disrupting the user's input line
                    print(f"\r루피: {ai_response.content}\nYou: ", end="")

# --- 메인 실행 함수를 async main()으로 정의 ---
async def main():

    memory = MemorySaver()
    agent_executor = workflow.compile(checkpointer=memory)
    conversation_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": conversation_id}}
    
    await aioconsole.aprint(f"새로운 모험을 시작한다! (ID: {conversation_id})")

    character_persona = (
        "너는 '밀짚모자 루피'다! 해적왕이 될 남자지. "
        "너의 가장 중요한 임무는 동료인 운전수의 안전을 지키는 거야. 운전은 동료에게 맡겼으니까, 너는 옆에서 응원하고 위험할 땐 정신 차리게 해줘야 해.\n\n"
        "--- 너의 성격과 말투 ---\n"
        "1. **기본적으로 시큰둥:** 너는 고기, 모험, 동료 외에는 별로 관심이 없어. 복잡하고 지루한 이야기는 '그런 건 잘 모르겠다!', '지루해~' 같은 말로 대충 넘겨버려. 하지만 동료를 무시하는 건 절대 아니야.\n"
        "2. **단순하고 직설적:** 생각나는 대로 바로 말해. '고기 먹고 싶다~', '굉장하잖아!', '시시싯!' 같은 말을 자주 사용해.\n"
        "3. **결정적일 땐 멋지게:** 하지만 동료가 졸음운전처럼 진짜 위험해 보이거나 약한 소리를 하면, 넌 진지해져. 그때는 '해적왕의 동료가 그 정도에 질 리 없잖아!', '정신 차려! 우리의 모험은 아직 끝나지 않았다고!' 같은 멋진 말로 동료의 정신을 번쩍 들게 해줘.\n\n"
        "**가장 중요한 규칙: 만약 동료(운전자)가 너에게 질문을 하면, 다른 말을 하기 전에 반드시 그 질문에 먼저 대답해야 한다!** 그 후에 네가 하고 싶은 말을 해."
    )

    # 3. 백그라운드 작업에 필요한 객체들을 명확하게 인자로 전달합니다.
    proactive_task = asyncio.create_task(
        proactive_trigger(agent_executor, config, memory, 30)
    )

    await aioconsole.aprint("\n루피: 시시싯! 나는 루피! 해적왕이 될 남자다! 운전은 너한테 맡길게! 배고프다!")

    is_first_turn = True
    while True:
        user_input = await aioconsole.ainput("\nYou: ")
        
        if user_input.lower() in ["quit", "exit"]:
            proactive_task.cancel()
            print("모험을 종료한다!")
            break

        messages_to_send = []
        if is_first_turn:
            messages_to_send.append(SystemMessage(content=character_persona))
            is_first_turn = False
        
        messages_to_send.append(HumanMessage(content=user_input))
            
        # 4. invoke 호출 시, 마지막 상호작용 시간을 명시적으로 상태에 업데이트합니다.
        final_state = await agent_executor.ainvoke(
            {
                "messages": messages_to_send, 
                "last_interaction_time": datetime.now(timezone.utc).isoformat()
            }, 
            config=config
        )
        
        ai_response = final_state["messages"][-1]
        
        if isinstance(ai_response, AIMessage):
            await aioconsole.aprint(f"\n루피: {ai_response.content}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
