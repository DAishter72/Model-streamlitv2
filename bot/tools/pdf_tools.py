import os
from typing import List
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader


@tool
def load_pdf_document(file_path: str) -> str:
    """Carga y extrae texto de un documento PDF. Para analizar el contenido de archivos PDF subidos por el usuario."""
    try:
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            return f"Error: El archivo {file_path} no existe en el servidor."

        # Verificar que es un archivo PDF
        if not file_path.lower().endswith('.pdf'):
            return f"Error: {file_path} no es un archivo PDF válido."

        # Cargar el documento con PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Extraer y combinar el texto
        text = "\n\n".join([doc.page_content for doc in documents])

        return text

    except Exception as e:
        return f"Error al procesar el PDF {file_path}: {str(e)}"


@tool
def analyze_uploaded_documents(document_paths: List[str]) -> str:
    """Analiza múltiples documentos PDF subidos por el usuario.
    Extrae y combina su contenido para su análisis."""
    if not document_paths:
        return "No hay documentos para analizar."

    analysis_results = []

    for doc_path in document_paths:
        if not os.path.exists(doc_path):
            analysis_results.append(f"El documento {doc_path} no existe.")
            continue

        doc_name = os.path.basename(doc_path)

        if doc_path.lower().endswith('.pdf'):
            content = load_pdf_document(doc_path)
            analysis_results.append(
                f"CONTENIDO DEL DOCUMENTO '{doc_name}':\n{content}\n")
        else:
            analysis_results.append(
                f"DOCUMENTO '{doc_name}': Tipo de archivo no compatible. Solo se pueden analizar PDFs.\n")

    return "\n".join(analysis_results)


@tool
def list_uploaded_documents(upload_dir: str = "uploaded_documents") -> List[str]:
    """Lista todos los documentos disponibles en el directorio de uploads."""
    if not os.path.exists(upload_dir):
        return []

    documents = []
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            documents.append(file_path)

    return documents


# Agregar estas tools a tu agente
tools = [
    load_pdf_document,
    analyze_uploaded_documents,
    list_uploaded_documents,
]
