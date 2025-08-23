import uuid
from utils import get_today_str, generate_luffy_audio
from datetime import datetime, timezone
import os
import asyncio

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# proactive_trigger와 memorySaver는 외부 모듈에서 가져온다고 가정
from sleepy_agent.analize_drowiness_agent import character_persona, agent_executor, proactive_trigger, MemorySaver, workflow


app = FastAPI()

# 생성된 MP3 파일을 외부에서 접근할 수 있도록 '/static' 경로를 열어줍니다.
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    # 클라이언트의 도메인을 여기에 추가하세요.
    # 만약 모든 출처를 허용하고 싶다면 "*"을 사용합니다.
    "https://970008f63656.ngrok-free.app/",
    "https://56a2c5f17dd3.ngrok-free.app/",
    "https://db7d300a3c9e.ngrok-free.app/",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


async def run_proactive_check(session_id: str, agent_executor, memory):
    """특정 세션에 대해 주기적으로 침묵을 감지하고 LangGraph를 호출하는 함수"""
    pass
    # config = {"configurable": {"thread_id": session_id}}
    # print(f"Proactive check started for session: {session_id}")
    
    # # 첫 번째 체크포인트를 기다리는 시간을 제한하고, 더 안전한 방식으로 체크
    # max_wait_time = 60  # 최대 60초 대기
    # wait_count = 0
    
    # while wait_count < max_wait_time:
    #     try:
    #         checkpoint = memory.get(config)
    #         if checkpoint:
    #             print(f"First checkpoint found for session: {session_id}")
    #             break
    #         print(f"Waiting for first checkpoint... ({wait_count}s)")
    #         await asyncio.sleep(1)
    #         wait_count += 1
    #     except Exception as e:
    #         print(f"Error getting checkpoint for session {session_id}: {e}")
    #         await asyncio.sleep(1)
    #         wait_count += 1
    
    # if wait_count >= max_wait_time:
    #     print(f"Timeout waiting for first checkpoint for session: {session_id}")
    #     return  # 타임아웃 시 함수 종료
    
    # # 주기적인 침묵 감지 루프
    # while True:
    #     try:
    #         # 30초 대기 (침묵 감지 주기)
    #         await asyncio.sleep(30)
    #         print(f"Checking silence for session: {session_id}")
            
    #         # 해당 세션의 상태를 가져옴
    #         checkpoint = memory.get(config)
            
    #         if checkpoint and "channel_values" in checkpoint:
    #             channel_values = checkpoint["channel_values"]
                
    #             # last_interaction_time 확인
    #             if "last_interaction_time" in channel_values:
    #                 last_time_str = channel_values["last_interaction_time"]
    #                 last_time = datetime.fromisoformat(last_time_str)
                    
    #                 # 침묵 시간을 확인 (30초 이상)
    #                 silence_duration = (datetime.now(timezone.utc) - last_time).total_seconds()
    #                 print(f"Session {session_id} - Silence duration: {silence_duration}s")
                    
    #                 if silence_duration > 30:
    #                     print(f"\n!!! Session {session_id}: 30초 이상 응답 없음. 루피가 먼저 말을 겁니다. !!!")
                        
    #                     # 프로액티브 메시지 발송
    #                     final_state = await agent_executor.ainvoke(
    #                         {
    #                             "messages": [HumanMessage(content="[USER_IS_SILENT]")],
    #                             "last_interaction_time": datetime.now(timezone.utc).isoformat()
    #                         },
    #                         config=config
    #                     )
                        
    #                     ai_response = final_state["messages"][-1]
    #                     if isinstance(ai_response, AIMessage):
    #                         print(f"루피: {ai_response.content}")
                            
    #                         # 필요하다면 여기서 오디오 생성 및 클라이언트에 알림 로직 추가 가능
    #             else:
    #                 print(f"No last_interaction_time found for session: {session_id}")
    #         else:
    #             print(f"No valid checkpoint found for session: {session_id}")

    #     except asyncio.CancelledError:
    #         print(f"Proactive task for session {session_id} has been cancelled.")
    #         break
    #     except Exception as e:
    #         print(f"Error in proactive task for session {session_id}: {e}")
    #         await asyncio.sleep(5)  # 오류 발생 시 잠시 대기 후 재시도


# 프론트엔드에서 요청 시 보낼 데이터의 형식을 정의합니다.
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    is_first_turn: bool = False


@app.on_event("startup")
async def startup_event():

    app.state.memory_saver = MemorySaver()
    app.state.agent_executor = workflow.compile(checkpointer=app.state.memory_saver)

    app.state.agent_executor = agent_executor.with_config(
        {"configurable": {"checkpointer": app.state.memory_saver}}
    )
    # 각 세션별 proactive task를 관리할 딕셔너리
    app.state.proactive_tasks = {}

@app.on_event("shutdown")
async def shutdown_event():
    """
    애플리케이션 종료 시 실행 중인 모든 proactive 태스크를 안전하게 취소합니다.
    """
    print("Shutting down proactive tasks...")
    for session_id, task in app.state.proactive_tasks.items():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    print("All proactive tasks have been cancelled.")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    STT로 변환된 텍스트(message)와 대화 ID(session_id)를 받아
    AI의 응답 텍스트와 MP3 파일의 공개 URL을 반환합니다.
    """
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    # LangGraph에 보낼 메시지 목록 생성
    messages_to_send = []
    if request.is_first_turn:
        messages_to_send.append(SystemMessage(content=character_persona))
    
    messages_to_send.append(HumanMessage(content=request.message))
    
    # agent_executor 호출 및 메모리에 저장
    final_state = await app.state.agent_executor.ainvoke(
        {
            "messages": messages_to_send, 
            "last_interaction_time": datetime.now(timezone.utc).isoformat()
        },
        config=config
    )
    
    # 💡 메시지 처리 완료 후에 proactive task 시작
    # if session_id not in app.state.proactive_tasks:
    #     # 잠시 대기 후 proactive task 시작 (체크포인트가 확실히 저장되도록)
    #     await asyncio.sleep(0.1)
        
    #     proactive_task = asyncio.create_task(
    #         run_proactive_check(
    #             session_id=session_id,
    #             agent_executor=app.state.agent_executor, 
    #             memory=app.state.memory_saver, 
    #         )
    #     )
    #     app.state.proactive_tasks[session_id] = proactive_task
    #     print(f"Proactive task created for session: {session_id}")
    
    # 최종 AI 응답 텍스트 추출
    ai_response = final_state["messages"][-1]
    ai_text_content = ""
    if isinstance(ai_response, AIMessage):
        ai_text_content = ai_response.content
    
    # ElevenLabs를 통해 MP3 파일 생성 (generate_luffy_audio는 utils.py에 정의)
    audio_filename = await generate_luffy_audio(ai_text_content)

    # # 프론트엔드에 전달된 ngrok 주소를 사용하여 전체 URL 생성
    # # base_url = "https://970008f63656.ngrok-free.app"
    # base_url = "https://56a2c5f17dd3.ngrok-free.app"
    # audio_url = f"{base_url}/static/{audio_filename}" if audio_filename else None

    # # 최종 결과 반환
    # return {
    #     "session_id": session_id,
    #     "ai_response": ai_text_content,
    #     "audio_url": audio_url
    # }

    if audio_filename:
        file_path = os.path.join("static", audio_filename)
        return FileResponse(file_path, media_type="audio/mpeg")
        
    # 오디오 생성에 실패한 경우
    return {"ai_response": ai_text_content, "audio_url": None}
