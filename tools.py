from re import search
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import json

def save_to_txt(data: str, filename: str = "research_output.txt"):
    """
    Guarda los resultados de investigación en un archivo de texto.
    
    Args:
        data: Los datos de investigación a guardar (puede ser texto, JSON, o cualquier formato)
        filename: Nombre del archivo donde guardar (por defecto: research_output.txt)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Formatear el contenido para guardar
    if isinstance(data, dict):
        # Si es un diccionario, formatearlo de manera legible
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n"
        for key, value in data.items():
            if key == "summary":
                formatted_text += f"{key.upper()}:\n{value}\n\n"
            elif key == "sources":
                formatted_text += f"{key.upper()}:\n"
                for i, source in enumerate(value, 1):
                    formatted_text += f"{i}. {source}\n"
                formatted_text += "\n"
            elif key == "source_links":
                formatted_text += f"{key.upper()}:\n"
                for i, link in enumerate(value, 1):
                    formatted_text += f"{i}. {link}\n"
                formatted_text += "\n"
            elif key == "tools_used":
                formatted_text += f"{key.upper()}: {', '.join(value)}\n\n"
            else:
                formatted_text += f"{key.upper()}: {value}\n\n"
    else:
        # Si es texto plano
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves research results to a text file. Use this tool after completing research to store the findings. Input should be the research data to save.",
)

# Búsqueda web más detallada con captura de enlaces
search = DuckDuckGoSearchRun()
def search_with_links(query: str) -> str:
    """Busca información y captura los enlaces de las fuentes"""
    try:
        # Usar DuckDuckGo para obtener resultados con enlaces
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            
        if results:
            # Formatear resultados con enlaces
            formatted_results = []
            for result in results:
                title = result.get('title', 'Sin título')
                link = result.get('link', 'Sin enlace')
                snippet = result.get('body', 'Sin descripción')
                formatted_results.append(f"Título: {title}\nEnlace: {link}\nDescripción: {snippet}\n")
            
            return "\n".join(formatted_results)
        else:
            return search.run(query)
    except Exception as e:
        # Fallback al método original si hay error
        return search.run(query)

search_tool = Tool(
    name="search",
    func=search_with_links,
    description="Search the web for current and detailed information. Use this to find recent articles, studies, news, and comprehensive information about any topic. Returns results with links to sources."
)

# Búsqueda específica para noticias con enlaces
def search_news_with_links(query: str) -> str:
    """Busca noticias recientes sobre el tema con enlaces"""
    news_query = f"noticias recientes {query}"
    return search_with_links(news_query)

news_tool = Tool(
    name="search_news",
    func=search_news_with_links,
    description="Search for recent news and current events related to the topic. Use this to find the latest developments and news articles. Returns results with links to sources."
)

# Búsqueda específica para estudios académicos con enlaces
def search_academic_with_links(query: str) -> str:
    """Busca estudios académicos y papers sobre el tema con enlaces"""
    academic_query = f"estudios académicos investigación papers {query}"
    return search_with_links(academic_query)

academic_tool = Tool(
    name="search_academic",
    func=search_academic_with_links,
    description="Search for academic studies, research papers, and scholarly articles about the topic. Use this to find scientific and academic information. Returns results with links to sources."
)

# Wikipedia con más resultados y enlaces
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
def wiki_with_links(query: str) -> str:
    """Busca en Wikipedia y devuelve información con enlaces"""
    try:
        # Usar la API de Wikipedia directamente para obtener enlaces
        import wikipedia
        
        # Buscar páginas relacionadas
        search_results = wikipedia.search(query, results=3)
        
        formatted_results = []
        for page_title in search_results:
            try:
                page = wikipedia.page(page_title)
                summary = wikipedia.summary(page_title, sentences=3)
                url = page.url
                formatted_results.append(f"Página: {page_title}\nEnlace: {url}\nResumen: {summary}\n")
            except:
                continue
        
        if formatted_results:
            return "\n".join(formatted_results)
        else:
            # Fallback al método original
            return wiki_tool.run(query)
    except Exception as e:
        # Fallback al método original si hay error
        return wiki_tool.run(query)

wiki_tool = Tool(
    name="wikipedia",
    func=wiki_with_links,
    description="Search Wikipedia for reference information and historical context. Returns results with links to Wikipedia pages."
)