from ast import Try
from pydoc import text
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """Eres un asistente de investigación que ayudará a generar un documento de investigación.
            Responde siempre en el idioma del usuario.
            Responde la consulta del usuario y usa las herramientas necesarias.
            Envuelve la salida en este formato y no proporciones otro texto\n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


agent = create_tool_calling_agent(
    llm = llm, 
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[], verbose=False)
raw_response = agent_executor.invoke({"query": "What is the capital of France?"})
""" print(raw_response) """

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response.sources)
except Exception as e:
    print(f"Error al procesar la respuesta: {e}", "Raw response", raw_response)
    #print(raw_response.get("output")[0]["text"])