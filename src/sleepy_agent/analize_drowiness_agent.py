from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from sleepy_agent.state_agent import LinguisticFatigueAnalysis
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=API_KEY,
    temperature=0,
    convert_system_message_to_human=True,
)

llm = model.with_structured_output(LinguisticFatigueAnalysis)


user_text = "네... 그냥... 운전 중... 잘 모르겠어요."

# 4. LLM에게 분석을 지시하는 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 운전자의 발화를 듣고 피로도를 분석하는 전문가입니다. "
     "주어진 텍스트에서 관찰되는 언어적 특징과, 그것이 왜 피로의 신호인지 근거를 들어 분석하고, "
     "제공된 LinguisticFatigueAnalysis 형식으로 출력하세요."),
    ("human", f"이 운전자의 발화를 분석해 주세요: '{user_text}'")
])

# 5. 체인 실행
chain = prompt | llm
result = chain.invoke({})

print(result)