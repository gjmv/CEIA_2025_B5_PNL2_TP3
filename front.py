# ========================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ========================================
from agente import (Agent)

import streamlit as st           # Framework para crear aplicaciones web interactivas
import os                        # Para acceso a variables de entorno
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_ollama import ChatOllama

load_dotenv() # Cargar variables de entorno desde el archivo .env

def main():

    # ========================================
    # CONFIGURACIÓN INICIAL Y AUTENTICACIÓN
    # ========================================
    
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        st.error("⚠️ PINECONE_API_KEY no está configurada en las variables de entorno")
        st.info("💡 Configura tu clave API: export PINECONE_API_KEY='tu-clave-aqui'")
        st.stop()  # Detener la ejecución si no hay clave API

    index_name = os.environ.get('PINECONE_INDEX_NAME') or 'ceia-2025-b5-pnl2-tp3'

    def load_or_create_from_session(key, default_value):
        """ Auxiliar para crear variables si no existen y mantenerlas en sesion """
        # Puede que no sea la mejor opción, pero mejora la performance al enviar las consultas
        # ya que no se regeneran las variables cada vez que se recarga la página
        if key not in st.session_state:
            st.session_state[key] = default_value()
        return st.session_state[key]
    
    ### EMBEDDINGS
    embedding_model = load_or_create_from_session("embedding_model",  lambda: HuggingFaceEmbeddings(model_name="all-mpnet-base-v2"))
    # Modelo
    model_name = os.environ.get('MODEL_NAME') or 'llama3:8b'
    model = load_or_create_from_session("model", lambda: ChatOllama(model=model_name))

    def create_people_names():
        # Connect to Pinecone DB
        pc=Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        namespaces = index_stats.namespaces
        people_names = []
        for namespace_name in namespaces:
            people_names.append(namespace_name)
            print(f"Persona disponible: {namespace_name}")
        return people_names

    people_names = load_or_create_from_session("people_names", create_people_names)

    # ========================================
    # CONFIGURACIÓN DE LA INTERFAZ PRINCIPAL
    # ========================================

    # Configurar el título y descripción de la aplicación
    st.title("🤖 Agente CEIA para consultar CVs de personas.")
    st.markdown(f"""
    **¡Bienvenido al agente CEIA - PNL2 - TP3!** 
    
    Este agente utiliza:
    - 🔄 **Modelo {model_name} (local)**: Instanciado mediante Ollama.
    - ⚙️ **Pinecone**: Almacenamiento de documentos para la búsqueda de respuestas
    - 🚀 **Powered by Ollama**: Integración con modelos locales
    """)

    # ========================================
    # PANEL DE CONFIGURACIÓN LATERAL
    # ========================================
    
    # Custom CSS to modify sidebar width
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the desired width here
        }
        </style>
        """,
        unsafe_allow_html=True)
    st.sidebar.title('⚙️ Características del Agente')
    st.sidebar.markdown("---")
    
    st.sidebar.info(f"Modelo {model_name}")
    
    # Input para el prompt del sistema - Define la personalidad y comportamiento del bot
    st.sidebar.subheader("🎭 CVs disponibles")
    st.sidebar.text_area(
        "CVs disponibles:",
        value="\n".join(people_names),
        disabled=True,
        help="CVs disponibles para consultar."
    )

    # Botón para limpiar y recargar todas las variables
    if st.sidebar.button("🗑️ Limpiar y reiniciar"):
        st.session_state = {}
        st.sidebar.success("✅ Reiniciado")
        st.rerun()  # Recargar la aplicación
    
    # ========================================
    # INTERFAZ DE ENTRADA DEL USUARIO
    # ========================================
    
    # Crear el campo de entrada para las preguntas del usuario
    st.markdown("### 💬 Haz tu pregunta:")
    user_question = st.text_input(
        "Escribe tu mensaje aquí:",
        placeholder="Por ejemplo: Que habilidades tiene Rob Otto?",
        label_visibility="collapsed",
        key="user_question"
    )


    # ========================================
    # CONFIGURACIÓN DEL MODELO DE LENGUAJE
    # ========================================
    
    # Inicializar el cliente de ChatGroq con las configuraciones seleccionadas
    try:
        agent = load_or_create_from_session("agent", lambda: Agent(
            people_names=people_names,
            default_person="Gustavo Vinas" if "Gustavo Vinas" in people_names else people_names[0],
            model=model,
            embedding_model=embedding_model,
            index_name=index_name
        ))

        st.sidebar.success("✅ Agente conectado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al conectar con el agente: {str(e)}")
        st.stop()

    # ========================================
    # PROCESAMIENTO DE LA PREGUNTA Y RESPUESTA
    # ========================================

    # Si el usuario ha hecho una pregunta,
    if user_question and user_question.strip():

        # Mostrar indicador de carga mientras se procesa
        with st.spinner('🤔 El agente está pensando...'):
            
            try:
                # ========================================
                # GENERACIÓN DE LA RESPUESTA
                # ========================================
                response = agent.ask(user_question)
 
                # ========================================
                # MOSTRAR LA CONVERSACIÓN
                # ========================================
                
                # Mostrar la respuesta actual destacada
                st.markdown("### 🤖 Respuesta:")
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                    {response}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                raise e
                # Manejo de errores durante el procesamiento
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                st.info("💡 Verifica tu conexión a internet y la configuración de la API")


    # ========================================
    # INFORMACIÓN ADICIONAL PARA ESTUDIANTES
    # ========================================
    
    # Panel expandible con información educativa
    with st.expander("📚 Información Técnica"):
        st.markdown("""
        **¿Cómo funciona este agente?**
        
        1. **Templates de Prompts**: Estructura los mensajes de manera consistente.
        2. **Router**: Extrae la o las personas a las que se refiere el usuario. Implementado mediante un LLM.
        3. **Agentes**: Creados dinámicamente en función de los individuos disponibles para la conversación.
        4. **Agregador**: Une la información de los agentes para generar una respuesta coherente.
        
        **Conceptos Clave:**
        - **Integración Ollama**: Acceso rápido a modelos de lenguaje modernos.
        - **System Prompt**: Define la personalidad del chatbot.
        - **Token Limits**: Al utilizar modelos locales, no costos asociados al consumo de tokens.
        
        **Arquitectura del Sistema:**
        ```
        Usuario → Streamlit → LangChain → Ollama → LLM (Router → Agentes → Agregador → Respuesta).
        ```
        """)
    
    # Pie de página con información del curso
    st.markdown("---")
    st.markdown("**📖 CEIA - 2025 - B5 - PNL2 - TP3** | Trabajo Práctico 3 - Procesamiento del Lenguaje Natural 2")


if __name__ == "__main__":
    # Punto de entrada de la aplicación
    # Solo ejecutar main() si este archivo se ejecuta directamente
    main()
