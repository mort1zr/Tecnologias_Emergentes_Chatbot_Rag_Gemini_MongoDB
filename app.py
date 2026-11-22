import streamlit as st
import pymongo
from google import genai
from google.genai import types

# ... pega aquí el resto del código de app.py ...

import streamlit as st
import pymongo
from google import genai
from google.genai import types

st.set_page_config(page_title="Chatbot RAG", page_icon="🤖")
st.title("🤖 Chatbot RAG sobre PDF")

# Paso 8: secretos
GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

# Paso 9: clientes en caché
@st.cache_resource
def get_clientes():
    client_genai = genai.Client(api_key=GOOGLE_API_KEY)
    client_mongo = pymongo.MongoClient(MONGODB_URI)
    collection = client_mongo.pdf_embeddings_db.pdf_vectors
    return client_genai, collection

client_genai, collection = get_clientes()

# Paso 10: recuperación semántica
def buscar_similares(pregunta, k=5):
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=pregunta,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    vector = response.embeddings[0].values

    pipeline = [
        {"$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": vector,
            "numCandidates": 100,
            "limit": k
        }},
        {"$project": {
            "_id": 0,
            "texto": 1,
            "score": {"$meta": "vectorSearchScore"}
        }}
    ]
    return list(collection.aggregate(pipeline))

# Paso 11: generación de la respuesta
def generar_respuesta(pregunta, fragmentos):
    if not fragmentos:
        return "No encontré información relevante en el documento."
    contexto = "\n\n".join(f["texto"] for f in fragmentos)
    prompt = f"""Eres un asistente que responde preguntas sobre un documento.
Usa EXCLUSIVAMENTE el contexto proporcionado. Si la respuesta no está
en el contexto, di claramente que no encontraste esa información en el documento.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

RESPUESTA:"""
    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# Paso 12: interfaz de chat
k = st.sidebar.slider("Fragmentos a recuperar (k)", 1, 10, 5)

if "historial" not in st.session_state:
    st.session_state.historial = []

def mostrar_fragmentos(fragmentos):
    with st.expander("Fragmentos recuperados"):
        for f in fragmentos:
            st.markdown(f"**Score:** {f['score']:.4f}")
            st.write(f["texto"])
            st.divider()

for msg in st.session_state.historial:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("fragmentos"):
            mostrar_fragmentos(msg["fragmentos"])

pregunta = st.chat_input("Escribe tu pregunta sobre el documento")

if pregunta:
    st.session_state.historial.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            fragmentos = buscar_similares(pregunta, k)
            respuesta = generar_respuesta(pregunta, fragmentos)
        st.markdown(respuesta)
        mostrar_fragmentos(fragmentos)

    st.session_state.historial.append({
        "role": "assistant",
        "content": respuesta,
        "fragmentos": fragmentos
    })
