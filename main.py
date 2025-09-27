import os
from datetime import datetime
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from bot.tools.pdf_tools import load_pdf_document

load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Chatbot v7.0",
    page_icon="📚",
    layout="wide"
)

# Título de la aplicación
st.title("🤖 Chatbot con documentos")
st.markdown("Sube tus documentos y haz preguntas sobre su contenido")

# Función para crear directorio de documentos


def create_documents_dir():
    """Crea el directorio para almacenar documentos"""
    documents_dir = "uploaded_documents"
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
    return documents_dir


# Función para guardar archivo


def save_uploaded_file(uploaded_file, documents_dir):
    """Guarda el archivo subido en el servidor"""
    # Generar nombre único para el archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(documents_dir, filename)

    # Guardar el archivo
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path, filename


# Función para procesar documentos y crear vector store con Cohere


def process_documents(texts):
    """Procesa los textos y crea un vector store para búsqueda semántica usando Cohere"""
    if not texts:
        return None

    # Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(texts)

    # Crear embeddings con Cohere Embed v4.0
    embeddings = CohereEmbeddings(
        user_agent="my-app/1.0",
        model="embed-v4.0"
    )

    # Crear vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store


# Función para inicializar el chatbot con Cohere


def initialize_chatbot(vector_store):
    """Inicializa el chatbot con el vector store usando Cohere"""
    llm = ChatCohere(
        model="command-r-plus-08-2024",
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    return qa_chain


# Sidebar para gestión de documentos
with st.sidebar:

    st.header("📁 Gestión de Documentos")
    st.info("Sube documentos PDF para analizar")

    # Sección de subida de archivos en sidebar
    uploaded_files = st.file_uploader(
        "Selecciona documentos",
        type=['pdf'],
        accept_multiple_files=True,
        help="Puedes subir archivos PDF"
    )

    # Botón para procesar documentos
    if uploaded_files:
        if st.button("🔄 Procesar Documentos", use_container_width=True):
            with st.spinner("Procesando documentos..."):
                all_text = ""
                processed_files = []
                documents_dir = create_documents_dir()

                for uploaded_file in uploaded_files:
                    # Guardar archivo
                    file_path, filename = save_uploaded_file(
                        uploaded_file, documents_dir)
                    processed_files.append(filename)

                    # Extraer texto
                    text = load_pdf_document(file_path)
                    if text:
                        all_text += f"\n\n--- Documento: {filename} ---\n\n{text}"

                if all_text:
                    # Procesar documentos
                    vector_store = process_documents(all_text)

                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.documents_processed = True
                        st.session_state.uploaded_files = processed_files
                        st.success(
                            f"✅ {len(processed_files)} documento(s) procesado(s)!")
                    else:
                        st.error("Error al procesar los documentos")
                else:
                    st.error("No se pudo extraer texto de los documentos")

    # Sección de visualización de documentos en sidebar
    st.header("📋 Documentos Almacenados")

    if st.button("🔄 Actualizar Lista", use_container_width=True):
        st.rerun()

    documents_dir = create_documents_dir()
    if os.path.exists(documents_dir) and os.listdir(documents_dir):
        st.success(
            f"Documentos encontrados ({len(os.listdir(documents_dir))}):")

        for filename in os.listdir(documents_dir):
            file_path = os.path.join(documents_dir, filename)
            file_size = os.path.getsize(file_path) / 1024  # Tamaño en KB

            col1, col2, col3 = st.columns([0.7, 0.2, 0.1])
            with col1:
                st.write(f"**{filename}**")
                st.caption(f"{file_size:.1f} KB")

            with col2:
                # Mostrar si está cargado actualmente
                if filename in st.session_state.get('uploaded_files', []):
                    st.success("✓ Activo")

            with col3:
                # Botón para eliminar
                if st.button("🗑️", key=f"delete_{filename}"):
                    try:
                        os.remove(file_path)
                        # Remover de la lista de archivos cargados
                        if filename in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.remove(filename)
                        st.success(f"Documento eliminado")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al eliminar: {str(e)}")
    else:
        st.info("No hay documentos almacenados")


# Inicializar estado de la sesión
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sección del chatbot (área principal)
st.header("💬 Chatbot v7.0")

# Mostrar estado de documentos cargados
if st.session_state.uploaded_files:
    st.success(f"📊 Documentos activos: {len(st.session_state.uploaded_files)}")
    for file in st.session_state.uploaded_files:
        st.write(f"• {file}")

if st.session_state.documents_processed and st.session_state.vector_store:
    # Inicializar chatbot
    qa_chain = initialize_chatbot(st.session_state.vector_store)

    # Mostrar historial de chat
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input del usuario
    user_input = st.chat_input("Haz una pregunta sobre los documentos...")

    if user_input:
        # Agregar pregunta al historial
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Obtener respuesta
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    result = qa_chain({"query": user_input})
                    response = result["result"]

                    # Mostrar fuentes
                    if 'source_documents' in result and result['source_documents']:
                        sources = list(set([os.path.basename(doc.metadata.get(
                            'source', 'Desconocido')) for doc in result['source_documents']]))
                        response += f"\n\n📚 **Fuentes:** {', '.join(sources)}"

                    st.markdown(response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error al procesar la pregunta: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg})

else:
    if not st.session_state.documents_processed:
        st.info(
            "👆 Sube y procesa algunos documentos en la sidebar para empezar a chatear")
