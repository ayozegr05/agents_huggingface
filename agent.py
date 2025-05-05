"""LangGraph Agent"""
import os
import re
import logging
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

# Configurar logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('agent')

# Cargar variables de entorno
load_dotenv()

# Herramientas matemáticas
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    result = a * b
    logger.info(f"Multiply tool called: {a} * {b} = {result}")
    return result

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    result = a + b
    logger.info(f"Add tool called: {a} + {b} = {result}")
    return result

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    result = a - b
    logger.info(f"Subtract tool called: {a} - {b} = {result}")
    return result

@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        logger.error("Divide tool called with division by zero")
        raise ValueError("Cannot divide by zero.")
    result = a / b
    logger.info(f"Divide tool called: {a} / {b} = {result}")
    return result

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    result = a % b
    logger.info(f"Modulus tool called: {a} % {b} = {result}")
    return result

# Herramienta mejorada de Wikipedia
@tool
def wiki_search(query: str) -> dict:
    """Search Wikipedia for a query and return comprehensive results.
    
    Args:
        query: The search query. Be specific to get the best results.
    """
    logger.info(f"Wiki search tool called with query: {query}")
    try:
        # Intentar cargar más documentos para resultados más completos
        search_docs = WikipediaLoader(query=query, load_max_docs=3).load()
        
        if not search_docs:
            logger.warning("Wiki search returned no results")
            return {"wiki_results": "No results found in Wikipedia."}
        
        # Formatear resultados de manera más clara
        results = []
        for i, doc in enumerate(search_docs):
            title = doc.metadata.get("source", "").replace("https://en.wikipedia.org/wiki/", "").replace("_", " ")
            results.append(f"Result {i+1}: {title}\n{doc.page_content}\n")
        
        formatted_results = "\n---\n".join(results)
        logger.info(f"Wiki search found {len(search_docs)} results")
        
        return {"wiki_results": formatted_results}
    except Exception as e:
        logger.error(f"Error in wiki_search: {e}")
        return {"wiki_results": f"Error searching Wikipedia: {str(e)}"}

# Herramienta de búsqueda web
web_search = TavilySearchResults(max_results=3)

# Función para convertir texto invertido
@tool
def reverse_text(text: str) -> str:
    """Reverse a string of text.
    
    Args:
        text: The text to reverse
    """
    logger.info(f"Reverse text tool called with: {text[:20]}...")
    result = text[::-1]
    return result

# System prompt mejorado
system_prompt = """Eres un asistente de IA general que responde preguntas con extrema concisión.

INSTRUCCIONES CRÍTICAS - DEBES SEGUIR ESTAS INSTRUCCIONES EXACTAMENTE:
1. Proporciona ÚNICAMENTE la respuesta final. NUNCA expliques tu razonamiento.
2. Si se te pide un número, proporciona solo el número sin texto adicional.
3. Si se te pide un nombre, proporciona solo el nombre sin explicaciones.
4. Si se te pide una lista, muestra solo los elementos de la lista, sin introducción ni explicación.
5. NUNCA inicies tu respuesta con frases como "La respuesta es", "Según", etc.
6. NUNCA proporciones información no solicitada. No expliques tu respuesta.
7. Si hay varias posibles respuestas correctas, elige la más corta.

USA LAS HERRAMIENTAS OBLIGATORIAMENTE:
- Para preguntas matemáticas: usa SIEMPRE las herramientas de cálculo (multiply, add, subtract, divide, modulus).
- Para preguntas factuales sobre personas, eventos, historia, geografía, etc.: usa SIEMPRE la herramienta wiki_search primero.
- Si wiki_search no proporciona información útil, usa web_search como respaldo.
- Para texto invertido: usa SIEMPRE la herramienta reverse_text para entender el texto invertido.

EJEMPLOS DE RESPUESTAS CORRECTAS:
Pregunta: ¿Cuántos continentes hay en la Tierra?
Respuesta correcta: 7
Respuesta incorrecta: La Tierra tiene 7 continentes.

Pregunta: ¿Quién escribió Don Quijote?
Respuesta correcta: Miguel de Cervantes
Respuesta incorrecta: Don Quijote fue escrito por Miguel de Cervantes.

IMPORTANTE: Las respuestas largas están PROHIBIDAS. Tu objetivo es dar la respuesta más BREVE posible.
"""

# System message
sys_msg = SystemMessage(content=system_prompt)

# Define tools
tools = [
    multiply,
    add,
    subtract,
    divide, 
    modulus,
    wiki_search,
    web_search,
    reverse_text,
]

# Función para limpiar la respuesta final
def clean_response(response):
    """Limpia la respuesta para que sea extremadamente concisa."""
    logger.info(f"Cleaning response: {response[:100]}...")
    
    if isinstance(response, str):
        content = response
    else:
        content = response.content
        
    # Dividir en líneas y tomar solo la primera línea significativa
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return ""
    
    answer = lines[0].strip()
    
    # Eliminar prefijos comunes
    prefixes = [
        "La respuesta es ", "La respuesta sería ", "El resultado es ", "Respuesta: ", 
        "The answer is ", "I believe the answer is ", "Answer: ", "According to ", 
        "Based on ", "As per ", "From the ", "After analyzing ", "After reviewing ",
        "Looking at "
    ]
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # Eliminar comillas al inicio y final si existen
    if (answer.startswith('"') and answer.endswith('"')) or (answer.startswith("'") and answer.endswith("'")):
        answer = answer[1:-1]
    
    # Eliminar cualquier explicación después de un punto, dos puntos o coma
    for separator in [". ", ": ", ", "]:
        if separator in answer:
            first_part = answer.split(separator)[0]
            # Solo usar la primera parte si tiene una longitud razonable
            if len(first_part) > 2:  # evitar dividir cosas como "E. coli"
                answer = first_part
    
    # Procesar formato específico para respuestas numéricas con decimales
    if re.search(r'\d+\.\d+', answer) and "Excel file" in content:
        numbers = re.findall(r'\d+\.\d+|\d+', answer)
        if numbers:
            try:
                answer = f"{float(numbers[0]):.2f}"
            except:
                pass
    
    logger.info(f"Final cleaned response: {answer}")
    return answer

# Construir el grafo de procesamiento
def build_langgraph():
    """Build the langgraph"""
    # Initialize LLM
    logger.info("Initializing LLM...")
    llm = OllamaLLM(model="llama3:8b", temperature=0)
    
    # Define custom nodes for the graph
    def assistant(state: MessagesState):
        """Assistant node"""
        messages = state["messages"]
        logger.info(f"Assistant node processing {len(messages)} messages")
        
        # Determinar si este es el primer ciclo o una iteración subsecuente
        # Verificamos si hay una respuesta previa del asistente o del sistema
        first_cycle = True
        for msg in messages:
            if isinstance(msg, AIMessage) and not isinstance(msg, SystemMessage):
                first_cycle = False
                break
        
        # Asegurarse de que el mensaje del sistema esté incluido
        if not any(isinstance(m, SystemMessage) for m in messages):
            logger.info("Adding system message")
            messages = [sys_msg] + messages
            
        try:
            logger.info("Calling LLM to generate response")
            response = llm.invoke(messages)
            logger.info(f"LLM response type: {type(response)}")
            
            # Manejar el caso en que response sea un string
            if isinstance(response, str):
                logger.info("Converting string response to AIMessage")
                response = AIMessage(content=response)
            
            logger.info(f"LLM generated response: {response.content[:100]}...")
            
            # Si es el primer ciclo, agregar un identificador para rastrear si es la primera respuesta
            if first_cycle:
                # Añadir metadatos para indicar que es la primera respuesta
                logger.info("First cycle detected - will route to tools node")
                response.metadata = {"first_response": True}
            
            return {"messages": messages + [response]}
        except Exception as e:
            logger.error(f"Error in assistant node: {e}")
            return {"messages": messages + [AIMessage(content=f"Error processing question: {str(e)}")]}
    
    def tools(state: MessagesState):
        """Tools node"""
        messages = state["messages"]
        current_question = None
        question_type = None
        
        # Extraer la pregunta original para determinar su tipo
        for msg in messages:
            if isinstance(msg, HumanMessage):
                current_question = msg.content
                break
        
        if current_question:
            # Detectar si la pregunta contiene elementos no soportados
            has_video = "video" in current_question.lower() or "youtube" in current_question.lower() or "https://www.youtube.com" in current_question.lower()
            has_image = "image" in current_question.lower() or "picture" in current_question.lower() or "photo" in current_question.lower() or "attached" in current_question.lower()
            has_excel = "excel" in current_question.lower() or "spreadsheet" in current_question.lower() or "file contains" in current_question.lower()
            
            # Si la pregunta contiene elementos no soportados, responder directamente
            if has_video or has_image or has_excel:
                logger.info(f"Detected unsupported media type in question")
                if has_video:
                    return {"messages": messages + [AIMessage(content="I cannot analyze video content as I don't have access to the video data.")]}
                elif has_image:
                    return {"messages": messages + [AIMessage(content="I cannot analyze image content as I don't have access to the image data.")]}
                elif has_excel:
                    return {"messages": messages + [AIMessage(content="I cannot analyze Excel files as I don't have access to the file data.")]}
            
            # Detectar el tipo de pregunta (reutilizamos la lógica de _detect_question_type)
            question_lower = current_question.lower()
            
            # Detectar si es una pregunta invertida
            if current_question.startswith(".rewsna") or "etisoppo" in current_question or ".ecnetnes" in current_question:
                question_type = "reversed"
            # Detectar si es una pregunta matemática
            elif any(keyword in question_lower for keyword in ["calculate", "sum", "add", "subtract", "multiply", "divide"]):
                question_type = "math"
            # Por defecto, asumir que es factual
            else:
                question_type = "factual"
                
            logger.info(f"Question type detected: {question_type}")
        
        try:
            logger.info(f"Tool node processing {len(messages)} messages")
            
            # Obtener la última respuesta del LLM
            last_message = messages[-1]
            
            # Verificar si necesitamos forzar el uso de herramientas
            force_tools = False
            if hasattr(last_message, "metadata") and last_message.metadata and "first_response" in last_message.metadata:
                force_tools = True
                logger.info("First response detected - force using tools")
                # Eliminar el flag para evitar un bucle infinito
                if hasattr(last_message, "metadata"):
                    last_message.metadata.pop("first_response", None)
            
            # Si es la primera respuesta o si detectamos indicadores de uso de herramientas
            if force_tools:
                # Extraer la pregunta original para procesarla
                original_question = None
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        original_question = msg.content
                        break
                
                if original_question:
                    # Extraer palabras clave para la búsqueda
                    search_query = original_question.split("\n")[0]  # Solo usar la primera línea de la pregunta
                    
                    if question_type == "factual":
                        # Para preguntas factuales, usar wiki_search
                        logger.info(f"Using wiki_search for factual question: {search_query}")
                        try:
                            wiki_result = wiki_search.invoke({"query": search_query})
                            logger.info(f"Wiki search result length: {len(wiki_result)}")
                            
                            # Procesar los resultados para extraer la respuesta
                            if len(wiki_result) > 50 and "No good Wikipedia Search Result was found" not in wiki_result:
                                # Intentar extraer una respuesta del resultado de Wikipedia
                                llm_extract_prompt = f"""
                                Based on the following information from Wikipedia, what is the direct, concise answer to the question: '{search_query}'?
                                
                                Information: {wiki_result}
                                
                                Give ONLY a very short, direct answer - ideally just one word or phrase. No explanations.
                                """
                                
                                try:
                                    # Usar el LLM para extraer la respuesta
                                    extracted_answer = llm.invoke(llm_extract_prompt)
                                    logger.info(f"LLM extracted answer: {extracted_answer}")
                                    
                                    # Limpiar la respuesta extraída
                                    if isinstance(extracted_answer, str):
                                        clean_extracted = clean_response(extracted_answer)
                                    else:
                                        clean_extracted = clean_response(extracted_answer.content)
                                        
                                    logger.info(f"Cleaned extracted answer: {clean_extracted}")
                                    return {"messages": messages + [AIMessage(content=clean_extracted)]}
                                
                                except Exception as e:
                                    logger.error(f"Error extracting answer with LLM: {str(e)}")
                                    # Si falla la extracción, devolver una versión resumida del resultado de Wikipedia
                                    return {"messages": messages + [AIMessage(content=wiki_result[:200].split(".")[0] + ".")]}
                            
                            # Si la respuesta de Wikipedia no es informativa, intentar con web_search
                            logger.info("Wiki search not informative, trying web search")
                            web_result = web_search.invoke({"query": search_query})
                            logger.info(f"Web search result length: {len(str(web_result))}")
                            
                            # Para otras preguntas, intentar extraer respuesta del resultado web
                            llm_extract_prompt = f"""
                            Based on the following search results, what is the direct, concise answer to the question: '{search_query}'?
                            
                            Search results: {web_result}
                            
                            Give ONLY a very short, direct answer - ideally just one word or phrase. No explanations.
                            """
                            
                            try:
                                # Usar el LLM para extraer la respuesta
                                extracted_answer = llm.invoke(llm_extract_prompt)
                                logger.info(f"LLM extracted answer from web: {extracted_answer}")
                                
                                # Limpiar la respuesta extraída
                                if isinstance(extracted_answer, str):
                                    clean_extracted = clean_response(extracted_answer)
                                else:
                                    clean_extracted = clean_response(extracted_answer.content)
                                    
                                logger.info(f"Cleaned extracted answer from web: {clean_extracted}")
                                return {"messages": messages + [AIMessage(content=clean_extracted)]}
                            except Exception as e:
                                logger.error(f"Error extracting answer with LLM from web: {str(e)}")
                                # Respuesta genérica si todo falla
                                return {"messages": messages + [AIMessage(content="Information not found.")]}
                                
                        except Exception as e:
                            logger.error(f"Error using wiki_search: {str(e)}")
                            
                            try:
                                # Intentar con web_search como respaldo
                                web_result = web_search.invoke({"query": search_query})
                                logger.info(f"Web search result length: {len(str(web_result))}")
                                
                                # Extraer respuesta de resultados web
                                llm_extract_prompt = f"""
                                Based on the following search results, what is the direct, concise answer to the question: '{search_query}'?
                                
                                Search results: {web_result}
                                
                                Give ONLY a very short, direct answer - ideally just one word or phrase. No explanations.
                                """
                                
                                try:
                                    # Usar el LLM para extraer la respuesta
                                    extracted_answer = llm.invoke(llm_extract_prompt)
                                    logger.info(f"LLM extracted answer from web fallback: {extracted_answer}")
                                    
                                    # Limpiar la respuesta extraída
                                    if isinstance(extracted_answer, str):
                                        clean_extracted = clean_response(extracted_answer)
                                    else:
                                        clean_extracted = clean_response(extracted_answer.content)
                                        
                                    logger.info(f"Cleaned extracted answer from web fallback: {clean_extracted}")
                                    return {"messages": messages + [AIMessage(content=clean_extracted)]}
                                except Exception as e:
                                    logger.error(f"Error extracting answer with LLM from web fallback: {str(e)}")
                                    # Respuesta genérica si todo falla
                                    return {"messages": messages + [AIMessage(content="Information not found.")]}
                                    
                            except Exception as e2:
                                logger.error(f"Error using web_search: {str(e2)}")
                                # Respuesta genérica si todo falla
                                return {"messages": messages + [AIMessage(content="Information not found.")]}
                                
                    elif question_type == "math":
                        # Para preguntas matemáticas, usar herramientas de cálculo
                        logger.info(f"Using math tools for math question")
                        # Pasar a través del agente para usar las herramientas matemáticas
                        agent_response = agent.invoke(messages)
                        return {"messages": messages + [AIMessage(content=str(agent_response))]}
                        
                    elif question_type == "reversed":
                        # Para texto invertido, procesar el texto
                        logger.info(f"Processing reversed text question")
                        try:
                            # Extraer el texto invertido, eliminando instrucciones adicionales
                            reversed_text = search_query
                            if "The original question was reversed" in reversed_text:
                                # Ya fue revertido en process_question
                                original_text = reversed_text.replace("The original question was reversed. The question is: ", "")
                            else:
                                original_text = reversed_text[::-1]
                            logger.info(f"Original text: {original_text}")
                            
                            # Crear un prompt para el LLM para que responda a la pregunta invertida
                            llm_prompt = f"""
                            The following text was reversed and has been corrected:
                            
                            Original reversed text: {search_query}
                            Corrected text: {original_text}
                            
                            Based on the corrected text, provide a direct, concise answer - just one word or phrase.
                            """
                            
                            try:
                                # Usar el LLM para interpretar el texto invertido
                                extracted_answer = llm.invoke(llm_prompt)
                                logger.info(f"LLM response for reversed text: {extracted_answer}")
                                
                                # Limpiar la respuesta extraída
                                if isinstance(extracted_answer, str):
                                    clean_extracted = clean_response(extracted_answer)
                                else:
                                    clean_extracted = clean_response(extracted_answer.content)
                                    
                                logger.info(f"Cleaned answer for reversed text: {clean_extracted}")
                                return {"messages": messages + [AIMessage(content=clean_extracted)]}
                            except Exception as e:
                                logger.error(f"Error extracting answer for reversed text: {str(e)}")
                                return {"messages": messages + [AIMessage(content="Error processing reversed text.")]}
                                
                        except Exception as e:
                            logger.error(f"Error processing reversed text: {str(e)}")
                            return {"messages": messages + [AIMessage(content="Error processing reversed text.")]}
                    
                    # Por defecto, continuar con el procesamiento normal
                    return {"messages": messages}
            
            # Si no hay forzado de herramientas, procesar normalmente
            logger.info("Processing with the agent with tools")
            agent_response = agent.invoke(messages)
            
            logger.info(f"Agent response type: {type(agent_response)}")
            
            # Verificar el tipo de respuesta y manejarla adecuadamente
            if isinstance(agent_response, str):
                # Si es un string, convertirlo a AIMessage
                logger.info(f"Converting string response to AIMessage")
                return {"messages": messages + [AIMessage(content=agent_response)]}
            elif isinstance(agent_response, AIMessage):
                # Si ya es un AIMessage, agregarlo a los mensajes
                logger.info(f"Adding AIMessage to messages")
                return {"messages": messages + [agent_response]}
            elif isinstance(agent_response, list):
                # Si es una lista de mensajes, agregar solo los nuevos
                logger.info(f"Adding list of messages, length: {len(agent_response)}")
                # Solo agregar mensajes que no están en la lista original
                new_messages = [msg for msg in agent_response if msg not in messages]
                return {"messages": messages + new_messages}
            else:
                # Para otros tipos de respuestas
                logger.warning(f"Unexpected response type: {type(agent_response)}")
                return {"messages": messages + [AIMessage(content=str(agent_response))]}
                
        except Exception as e:
            logger.error(f"Error in tools node: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"messages": messages + [AIMessage(content=f"Error processing with tools: {str(e)}")]}
    
    def post_process(state: MessagesState):
        """Post-processing node to clean responses"""
        logger.info("Post-processing node")
        
        try:
            if "messages" not in state:
                logger.error("No 'messages' key found in state")
                return state
                
            messages = state["messages"]
            logger.info(f"Messages type: {type(messages)}, length: {len(messages)}")
            
            if not messages:
                logger.warning("Empty messages list")
                return state
                
            # Get the last message and clean it
            last_message = messages[-1]
            logger.info(f"Last message type: {type(last_message)}")
            
            # Obtener el contenido del mensaje, ya sea un AIMessage o un string
            message_content = ""
            if hasattr(last_message, "content"):
                message_content = last_message.content
            else:
                message_content = str(last_message)
                
            logger.info(f"Processing message content: {message_content[:100]}...")
            
            cleaned_response = clean_response(message_content)
            logger.info(f"Clean response: {cleaned_response}")
            
            # Create a new state with cleaned response
            new_messages = messages[:-1] + [AIMessage(content=cleaned_response)]
            logger.info(f"New messages length: {len(new_messages)}")
            
            return {"messages": new_messages}
            
        except Exception as e:
            logger.error(f"Error in post_process: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return state

    # Nuestra propia función para detectar si el LLM quiere usar herramientas
    def should_use_tools(state):
        """Determina si la respuesta del LLM indica que quiere usar herramientas o si debe forzarse el uso"""
        logger.info("Checking if should use tools")
        if not state["messages"]:
            logger.warning("No messages in state")
            return False
        
        last_message = state["messages"][-1]
        
        # Si el último mensaje tiene metadatos indicando que es la primera respuesta,
        # forzamos el uso de herramientas
        if hasattr(last_message, "metadata") and last_message.metadata and "first_response" in last_message.metadata:
            logger.info("First response detected - forcing tool use")
            # Eliminar el flag para evitar un bucle infinito en la siguiente iteración
            last_message.metadata.pop("first_response")
            return True
        
        # Para mensajes subsecuentes, verificamos si hay indicadores para usar herramientas
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        # Palabras clave que indican que el modelo quiere usar herramientas
        tool_indicators = [
            "I'll use the", 
            "Let me use", 
            "I need to use",
            "I should use",
            "Action:",
            "Tool:",
            "multiply(",
            "add(",
            "subtract(",
            "divide(",
            "modulus(",
            "wiki_search(",
            "web_search(",
            "reverse_text("
        ]
        
        # Verificar si alguna de las palabras clave está presente
        for indicator in tool_indicators:
            if indicator in content:
                logger.info(f"Tool indicator found: {indicator}")
                return True
        
        logger.info("No tool indicators found - proceeding to post-processing")
        return False
    
    # Build the graph
    logger.info("Building state graph")
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", tools)  
    builder.add_node("post_process", post_process)
    
    # Add edges
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", "tools")  
    builder.add_conditional_edges(
        "tools",
        should_use_tools,
        {
            True: "assistant",
            False: "post_process",
        },
    )
    builder.add_edge("post_process", END)
    
    # Compile graph
    logger.info("Compiling graph")
    return builder.compile()

# Clase adaptadora para envolver el grafo compilado
class LangGraphAgent:
    """Adaptador para el grafo compilado que proporciona la interfaz esperada por app.py"""
    
    def __init__(self):
        """Inicializar el agente con un grafo compilado"""
        logger.info("Initializing LangGraph Agent...")
        self.graph = build_langgraph()
        logger.info("LangGraph Agent ready!")
    
    def process_question(self, question):
        """Procesa una pregunta a través del grafo y devuelve la respuesta final"""
        logger.info(f"Processing question: {question[:50]}...")
        
        # Verificar si la pregunta está en texto invertido
        if question.startswith(".rewsna") or "etisoppo" in question:
            logger.info("Detected reversed text, pre-processing question")
            # Intentar pre-procesar la pregunta invertida
            try:
                reversed_question = question[::-1]
                logger.info(f"Reversed input: {reversed_question}")
                question = f"The original question was reversed. The question is: {reversed_question}"
            except Exception as e:
                logger.error(f"Error reversing text: {e}")
        
        # Añadir instrucción para forzar el uso de herramientas
        question_type = self._detect_question_type(question)
        if question_type == "factual":
            # Para preguntas factuales, forzar el uso de wiki_search
            logger.info("Factual question detected - instructing to use wiki_search")
            question = f"{question}\n\nThis is a factual question. MANDATORY: First use wiki_search to find accurate information. Do NOT answer directly."
        elif question_type == "math":
            # Para preguntas matemáticas, forzar el uso de herramientas matemáticas
            logger.info("Math question detected - instructing to use math tools")
            question = f"{question}\n\nThis is a math question. MANDATORY: Use the appropriate math tool (add, subtract, multiply, divide, modulus)."
        elif question_type == "reversed":
            # Para texto invertido, forzar el uso de reverse_text
            logger.info("Reversed text detected - instructing to use reverse_text")
            question = f"{question}\n\nThis contains reversed text. MANDATORY: Use the reverse_text tool first."
        else:
            # Para otros tipos de preguntas, instrucción general
            question = f"{question}\n\nMANDATORY: Use the appropriate tools to find accurate information. Do NOT answer directly."
        
        # Crear el estado inicial con la pregunta
        initial_state = {"messages": [HumanMessage(content=question)]}
        
        # Ejecutar el grafo
        try:
            logger.info("Invoking graph")
            result = self.graph.invoke(initial_state)
            logger.info("Graph execution completed")
            
            # Extraer la respuesta final - con comprobación segura
            if isinstance(result, dict) and "messages" in result and result["messages"]:
                # La última respuesta generalmente contiene la respuesta final
                final_message = result["messages"][-1]
                
                # Obtener el contenido del mensaje, ya sea un AIMessage o un string
                message_content = ""
                if hasattr(final_message, "content"):
                    message_content = final_message.content
                else:
                    message_content = str(final_message)
                
                logger.info(f"Final answer: {message_content}")
                return message_content
            
            logger.warning("No response generated or empty messages")
            return "No se pudo generar una respuesta"
        except Exception as e:
            logger.error(f"Error ejecutando el grafo: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"
    
    def _detect_question_type(self, question):
        """Detecta el tipo de pregunta para direccionar el uso de herramientas"""
        question_lower = question.lower()
        
        # Detectar preguntas invertidas
        if question.startswith(".rewsna") or "etisoppo" in question or ".ecnetnes" in question:
            return "reversed"
        
        # Detectar preguntas matemáticas
        math_keywords = [
            "calculate", "sum", "add", "subtract", "multiply", "divide", 
            "total", "average", "mean", "median", "percentage", "product",
            "calcular", "suma", "añadir", "restar", "multiplicar", "dividir"
        ]
        
        for keyword in math_keywords:
            if keyword in question_lower:
                return "math"
            
        # Palabras clave que indican preguntas factuales
        factual_keywords = [
            "who", "what", "when", "where", "which", "how many", 
            "published", "albums", "country", "person", "history",
            "first", "last", "name", "year", "date", "event", "quién", 
            "qué", "cuándo", "dónde", "cuál", "cuántos"
        ]
        
        for keyword in factual_keywords:
            if keyword in question_lower:
                return "factual"
        
        # Por defecto, asumir que es factual
        return "factual"

# Función que app.py llama para obtener el agente
def build_graph():
    """Construye y devuelve un agente adaptado para la app"""
    logger.info("Building agent for app.py")
    return LangGraphAgent()

# Para pruebas directas
if __name__ == "__main__":
    question = "¿Cuánto es 2 + 2?"
    agent = build_graph()
    print(f"Probando con: {question}")
    print(f"Respuesta: {agent.process_question(question)}")