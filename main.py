from ast import Try
from pydoc import text
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, news_tool, academic_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    key_points: list[str]
    current_developments: str
    academic_insights: str
    sources: list[str]
    source_links: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            """Eres un asistente de investigación experto que genera informes detallados y completos.
            Responde siempre en el idioma del usuario.
            
            INSTRUCCIONES IMPORTANTES:
            1. Realiza una investigación COMPLETA usando MÚLTIPLES herramientas en este orden:
               - Usa 'search' para información general y actual sobre el tema
               - Usa 'search_news' para noticias recientes y desarrollos actuales
               - Usa 'search_academic' para estudios académicos y papers científicos
               - Usa 'wikipedia' para información de referencia y contexto histórico
            
            2. Después de recopilar toda la información, SIEMPRE usa 'save_text_to_file' para guardar los resultados
            
            3. Genera un informe estructurado con:
               - Resumen ejecutivo detallado (mínimo 200 palabras)
               - 5-7 puntos clave principales
               - Desarrollos actuales y noticias recientes
               - Perspectivas académicas y científicas
               - Fuentes completas y diversas
               - ENLACES DIRECTOS a las fuentes consultadas
            
            4. IMPORTANTE: Extrae y guarda TODOS los enlaces (URLs) que encuentres en las búsquedas.
               Los enlaces deben incluirse en el campo 'source_links' del formato de salida.
            
            5. Envuelve la salida final en este formato: {format_instructions}
            
            IMPORTANTE: No te detengas hasta completar TODAS las búsquedas y generar el informe completo.
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool, news_tool, academic_tool]
agent = create_tool_calling_agent(
    llm = llm, 
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
query = input("What can i help you research?\n")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    print("\n" + "="*50)
    print("INFORME DE INVESTIGACIÓN COMPLETO")
    print("="*50)
    print(f"\nTEMA: {structured_response.topic}")
    print(f"\nRESUMEN EJECUTIVO:\n{structured_response.summary}")
    print(f"\nPUNTOS CLAVE:")
    for i, point in enumerate(structured_response.key_points, 1):
        print(f"{i}. {point}")
    print(f"\nDESARROLLOS ACTUALES:\n{structured_response.current_developments}")
    print(f"\nPERSPECTIVAS ACADÉMICAS:\n{structured_response.academic_insights}")
    print(f"\nFUENTES CONSULTADAS:")
    for i, source in enumerate(structured_response.sources, 1):
        print(f"{i}. {source}")
    print(f"\nENLACES DE FUENTES:")
    for i, link in enumerate(structured_response.source_links, 1):
        print(f"{i}. {link}")
    print(f"\nHERRAMIENTAS UTILIZADAS: {', '.join(structured_response.tools_used)}")
    print("="*50)
except Exception as e:
    print(f"Error al procesar la respuesta: {e}")
    print("Raw response:", raw_response)