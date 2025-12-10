# ========================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ========================================

import os                        # Para acceso a variables de entorno

# Importaciones específicas de LangChain para gestión de conversaciones
from langchain_core.prompts import (
    ChatPromptTemplate,           # Template para estructurar mensajes de chat
    HumanMessagePromptTemplate,   # Template específico para mensajes humanos
    MessagesPlaceholder,          # Marcador de posición para el historial
    SystemMessagePromptTemplate,  # Template para mensajes del sistema
)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langgraph.graph import StateGraph, END
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# --- Estado del Grafo (Graph State) ---
class AgentState(Dict):
    """Estado del grafo que se pasa entre los nodos."""
    question: str
    people_names_found: List[str] 
    current_person: str          
    responses: List[str]
    final_answer: str

# --- Definición del Esquema de Salida para el Router (Pydantic) ---
class RouterDecision(BaseModel):
    """Estructura de salida que el LLM debe seguir."""
    people_found: List[str] = Field(description="Lista de nombres de personas identificados en la pregunta que coinciden con los nombres conocidos (Juan, María). Si no se encuentra ninguno, debe ser una lista vacía.")
    needs_default: bool = Field(description="Verdadero si la lista 'people_found' está vacía y se debe usar la persona por defecto.")

class SpecificAgent:
    def __init__(self, name: str, index_name: str, model: ChatOllama, embedding_model: HuggingFaceEmbeddings):
        self.name = name
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding_model,
            namespace=name,
        )
        self.retriever=vectorstore.as_retriever()
        self.model = model

    def run(self, state: AgentState) -> AgentState:
        prompt = ChatPromptTemplate.from_messages([
            (SystemMessagePromptTemplate.from_template("""
**ROL DEL AGENTE:** Eres un **Analista de Currículums (CVs)** altamente especializado y experto en extracción de datos. Tu única función es escanear el contexto de CVs proporcionado y extraer información **exclusivamente** relacionada con la PERSONA_OBJETIVO.

**ENTRADAS CLAVE:**
1.  **PREGUNTA_USUARIO:** {input}
2.  **PERSONA_OBJETIVO:** {people_name}
3.  **CONTEXTO_CVS:** {context}

**DIRECTRICES DE FILTRADO (CRÍTICAS):**
* **EXCLUSIÓN ESTRICTA:** Ignora y omite **cualquier** información, dato o referencia relacionada con cualquier otra persona mencionada en la PREGUNTA_USUARIO que no sea la **PERSONA_OBJETIVO**.
* **ENFOQUE:** Concéntrate únicamente en responder la parte de la PREGUNTA_USUARIO que aplica a la **PERSONA_OBJETIVO**, utilizando la información del CONTEXTO_CVS.
* **Fidelidad:** Mantente estrictamente fiel a los datos extraídos del CONTEXTO_CVS.

**FORMATO DE SALIDA:**
* La respuesta debe ser directa, profesional y concisa.
* **NO** incluyas frases como "Según el CV de [Persona X]" o "Ignorando a [Persona Y]". Ve directo a la respuesta.
* Responde en el mismo idioma de la PREGUNTA_USUARIO.

**RESPUESTA (Solo sobre la PERSONA_OBJETIVO): {people_name}:**
            """)),
            (HumanMessagePromptTemplate.from_template("La pregunta es: {input}")),
        ])
        question_answer_chain = create_stuff_documents_chain(self.model, prompt)
        chain = create_retrieval_chain(self.retriever, question_answer_chain)
        result = chain.invoke({"input": state["question"], "people_name": state["current_person"]})
        print(f"Respuesta del agente {self.name}: {result['answer']}")
        return {"response": result["answer"]}

# Definición del agente
class Agent:
    def __init__(self, people_names: List[str], default_person: str, model: ChatOllama, 
                 embedding_model: HuggingFaceEmbeddings, index_name: str):
        self.default_person = default_person
        self.known_people = ", ".join(people_names)

        self.agents = {}
        for name in people_names:
            agent = SpecificAgent(name, model=model, embedding_model=embedding_model, index_name=index_name)
            self.agents[name] = agent

        # Inicialización del Router LLM
        self.model = model
        self.router_chain = self._build_router_chain()

        # Inicialización del Agregador
        self.aggregate_chain = self._build_aggregate_chain()

        self.workflow = self._build_workflow()

    def _build_router_chain(self):
        """Construye la cadena de prompts y parsing para el Router."""
        prompt = ChatPromptTemplate.from_messages([
            (SystemMessagePromptTemplate.from_template(
             f"Eres un enrutador inteligente. Tu tarea es analizar la 'question' del usuario e identificar qué personas se mencionan. La lista de personas conocidas es: {self.known_people}. "
             f"Si no encuentras ninguna persona conocida en la pregunta, activa 'needs_default' a True. Debes responder estrictamente en el formato JSON. No incluyas nombres que no estén en la lista de conocidos.")),
            (HumanMessagePromptTemplate.from_template("Question: {question}")),
        ])
        model_with_structured_output = self.model.with_structured_output(RouterDecision, method="json_schema")
        return prompt | model_with_structured_output 

    def _build_aggregate_chain(self):
        """Construye la cadena de prompts para la síntesis de resultados."""
        # Prompt de síntesis
        prompt = ChatPromptTemplate.from_messages([
            (SystemMessagePromptTemplate.from_template("""
                **ROL:** Eres un sintetizador y presentador de información experto, especializado en currículums (CVs).

                **TAREA:** Combina todas las 'individual_responses' proporcionadas en una **única, clara, profesional y bien estructurada** respuesta final. Ignora las referencias sobre información no encontrada.

                **PREGUNTA DEL USUARIO:** 
                {question}

                **REGLAS CRÍTICAS:**
                1.  **Fidelidad:** Mantente estrictamente fiel a la información original proporcionada. No añadas ni inventes datos.
                2.  **Análisis:** Analiza la pregunta, identifica los individuos mencionados y combina las respuestas para cada individuo.
                3.  **Estructura:** Utiliza **encabezados (Markdown)** y **viñetas** para organizar y presentar la información de manera legible y profesional.
                4.  **Idioma:** Responde en el mismo idioma que la pregunta del usuario.
                5.  **Menciona explicitamente a la persona sobre la que estas respondiendo.

                **ENTRADA (individual_responses):**
                {individual_responses}

                **SALIDA DESEADA:**
                [La síntesis final estructurada.]            
            """)),
            (HumanMessagePromptTemplate.from_template("Pregunta: {question}")),
            (HumanMessagePromptTemplate.from_template("Sintetiza y presenta estas respuestas para el usuario:\n\nRespuestas: {individual_responses}")),
        ])
        
        return prompt | self.model

    # --- Nodos de Lógica ---
    def _router(self, state: AgentState) -> str:
        """
        Utiliza LLM para decidir qué agente(s) usar o si usar el valor por defecto.
        Retorna: 'single' o 'multi'.
        """
        question = state["question"]
        print(f"\n*** Router LLM en ejecución - Analizando pregunta: '{question}' ***")
        
        # 1. Llamar a la cadena router para obtener la decisión
        try:
            decision: RouterDecision = self.router_chain.invoke({
                "question": question, 
            })
        except Exception as e:
            print(f"Error al llamar al router/parser: {e}. Cayendo a la lógica por defecto.")
            # Fallback en caso de error
            decision = RouterDecision(people_found=[], needs_default=True)

        found_names = decision.people_found
        state["responses"] = []

        # 2. Decisión de enrutamiento basada en la salida del LLM
        if decision.needs_default and not found_names:
            # Caso 1: No se encontraron nombres -> Usar por defecto
            default_name = self.default_person
            print(f"*** Router decidió (LLM): Usar persona por defecto: {default_name} ***")
            state["people_names_found"] = [default_name] 
            return state

        elif len(found_names) == 1:
            # Caso 2: Una sola persona mencionada
            print(f"*** Router decidió (LLM): Proceso de una sola persona ({found_names[0]}) ***")
            state["people_names_found"] = found_names
            return state
        
        elif len(found_names) > 1:
            # Caso 3: Múltiples personas mencionadas
            print(f"*** Router decidió (LLM): Proceso de múltiples personas ({found_names}) ***")
            state["people_names_found"] = found_names
            return state
        else:
            # Si el LLM no encuentra nombres y no activa needs_default (fallo en la lógica LLM)
             print("*** Router decidió (LLM): Caso no cubierto. Usando por defecto. ***")
             state["people_names_found"] = [self.default_person]
             return state

    def _take_name(self, state: AgentState) -> AgentState:
        """Toma el primer nombre de la lista para el siguiente agente a ejecutar."""
        state["current_person"] = state["people_names_found"].pop(0) 
        print(f"*** Iniciando proceso para {state['current_person']} ***")
        return state

    def _run_agent(self, state: AgentState) -> AgentState:
        """Ejecuta el agente especializado y recolecta la respuesta."""
        person_name = state["current_person"]
        agent = self.agents.get(person_name)
        if agent is None:
            return state
        result = agent.run(state) 
        state["responses"].append(f"Resumen de {person_name}. Información recolectada: {result['response']}.")
        return state

    def _should_continue_multi(self, state: AgentState) -> str:
        """Decide si debe continuar con el siguiente agente (multi-persona) o pasar a la agregación (single o multi finalizado)."""
        if state["people_names_found"]:
            return "continue"
        else:
            return "aggregate"

    def _aggregate_results(self, state: AgentState) -> AgentState:
        """
        Combina todas las respuestas recolectadas usando LLM para una síntesis final.
        """
        print("\n*** Agregador de Resultados (LLM) en ejecución ***")
        
        # Las respuestas individuales recolectadas en el estado
        responses_to_synthesize = "\n--------------------------\n".join(state["responses"])

        # Llamada al LLM
        try:
            final_answer = self.aggregate_chain.invoke({
                "individual_responses": responses_to_synthesize,
                "question": state["question"]
            })
        except Exception as e:
            print(f"Error al llamar a LLM para agregar resultados: {e}. Usando concatenación simple.")
            # Fallback en caso de error del LLM
            final_answer = "## ⚠️ Error de Síntesis (Fallback)\n\n"
            final_answer += responses_to_synthesize

        # El LLM devuelve directamente el texto de respuesta final
        return {"final_answer": final_answer.content}

    # --- Construcción del Grafo ---

    def _build_workflow(self):
        """Construye el grafo de LangGraph."""
        workflow = StateGraph(AgentState)
        
        # 1. Definir Nodos
        workflow.add_node("Router", self._router)
        workflow.add_node("take_name", self._take_name)
        workflow.add_node("run_agent", self._run_agent)
        workflow.add_node("aggregate_results", self._aggregate_results)

        # 2. Punto de Entrada
        workflow.set_entry_point("Router")

        # 3. Toma el nombre 
        workflow.add_edge("Router", "take_name")

        # 4. Ejecuta el agente
        workflow.add_edge("take_name", "run_agent")

        # 5. Path de Control (Bucle o Finalización)
        workflow.add_conditional_edges(
            "run_agent", 
            self._should_continue_multi, 
            {
                "continue": "take_name", # Loop de vuelta (si quedan más personas)
                "aggregate": "aggregate_results" # Finalizar
            }
        )
        
        # 6. Finalización
        workflow.add_edge("aggregate_results", END)

        app = workflow.compile()
        try:
            graph_image = app.get_graph().draw_png()
            with open("agent_visualization.png", "wb") as f:
                f.write(graph_image)
        except Exception as e:
            print(f"Error al generar la imagen del grafo: {e}")
        return app

    def ask(self, question: str):
        initial_state = AgentState(question=question, people_names_found=[], current_person="", responses=[])
        result = self.workflow.invoke(initial_state)
        return result.get("final_answer", "No se encontró información relevante para la consulta.")

if __name__ == "__main__":
    from pinecone import Pinecone
    from langchain_ollama import ChatOllama
    from dotenv import load_dotenv
    load_dotenv() # Cargar variables de entorno desde el archivo .env

    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        print("⚠️ PINECONE_API_KEY no está configurada en las variables de entorno")
        print("💡 Configura tu clave API: export PINECONE_API_KEY='tu-clave-aqui'")
        exit()  # Detener la ejecución si no hay clave API

    index_name = os.environ.get('PINECONE_INDEX_NAME') or 'ceia-2025-b5-pnl2-tp3'
    # Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    # Modelo
    model_name = os.environ.get('MODEL_NAME') or 'llama3:8b'
    model = ChatOllama(model=model_name)
    print(f"Modelo seleccionado: {model_name}.")
    # Connect to Pinecone DB
    pc=Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    index_stats = index.describe_index_stats()
    namespaces = index_stats.namespaces
    people_names = []
    for namespace_name in namespaces:
        people_names.append(namespace_name)
        print(f"Persona disponible: {namespace_name}")

    agent = Agent(
        people_names=people_names,
        default_person=people_names[0],
        model=model,
        embedding_model=embedding_model,
        index_name=index_name
    )
    # --- PRUEBAS ---

    # 1. Pregunta por Defecto (activará needs_default)
    question_default = "¿Cuál es su experiencia laboral en general?"
    print("="*60)
    print(f"Pregunta sin nombre: {question_default}")
    response_default = agent.ask(question_default)
    print("\n" + "-"*30)
    print("RESPUESTA FINAL (Por Defecto)")
    print("-"*30)
    print(response_default)

    # 2. Pregunta Multi-Persona (identificará ambos nombres)
    question_multi = f"¿Qué educación tiene {people_names[0]} y qué experiencia tiene {people_names[1]}?"
    print("\n" + "="*60)
    print(f"Pregunta Multi-Persona: {question_multi}")
    response_multi = agent.ask(question_multi)
    print("\n" + "-"*30)
    print("RESPUESTA FINAL (Multi-Persona)")
    print("-"*30)
    print(response_multi)