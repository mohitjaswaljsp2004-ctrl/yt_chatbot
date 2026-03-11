import streamlit as st
import os
import re
import json
from datetime import datetime
from yt_dlp import YoutubeDL
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# ---------------- LOAD ENV ---------------- #

load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------- STREAMLIT CONFIG ---------------- #

st.set_page_config(page_title="YouTube RAG Chatbot", layout="wide")
st.title("🎥 Chat with YouTube Video")

MEMORY_FILE = "videos_memory.json"


# ---------------- MEMORY FUNCTIONS ---------------- #

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)


def add_video_to_memory(video_url, transcript):
    memory = load_memory()

    for video in memory:
        if video["url"] == video_url:
            return

    memory.append({
        "url": video_url,
        "transcript_preview": transcript[:500],
        "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    save_memory(memory)


# ---------------- DOWNLOAD SUBTITLES ---------------- #

def download_subtitles(video_url):

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "outtmpl": "yt_text.%(ext)s",
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


# ---------------- VTT → TEXT ---------------- #

def vtt_to_text(path):

    text = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = re.sub(r"<[^>]+>", "", line).strip()

            if line and not line[0].isdigit() and "WEBVTT" not in line:
                text.append(line)

    return " ".join(text)


# ---------------- VECTOR STORE ---------------- #

@st.cache_resource(show_spinner=False)
def build_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


# ---------------- SIDEBAR MEMORY ---------------- #

st.sidebar.header("📚 Previously Searched Videos")

memory = load_memory()


def get_video_info(url):
    try:
        ydl_opts = {"quiet": True, "skip_download": True}

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        title = info.get("title", "No title")
        thumbnail = info.get("thumbnail", "")

        return title, thumbnail

    except:
        return "Unknown title", ""


if memory:

    for i, video in enumerate(reversed(memory)):

        title, thumbnail = get_video_info(video["url"])

        if thumbnail:
            st.sidebar.image(thumbnail, use_container_width=True)

        if st.sidebar.button(title, key=f"video_button_{i}"):
            st.session_state.selected_video = video["url"]

        st.sidebar.caption(video["date_added"])
        st.sidebar.write("---")

else:
    st.sidebar.info("No videos searched yet.")


# ---------------- MAIN UI ---------------- #

video_url = st.text_input(
    "📎 Enter YouTube Video URL",
    value=st.session_state.get("selected_video", "")
)


if st.button("📥 Process Video", key="process_video"):

    if not video_url:
        st.warning("Please enter a YouTube URL.")

    else:

        with st.spinner("Downloading subtitles..."):
            download_subtitles(video_url)

        vtt_file = "yt_text.en.vtt"

        if not os.path.exists(vtt_file):
            st.error("❌ English subtitles not found.")

        else:

            with st.spinner("Processing transcript & embeddings..."):

                text = vtt_to_text(vtt_file)

                os.remove(vtt_file)

                vector_store = build_vector_store(text)

                st.session_state.retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                st.session_state.transcript = text

                add_video_to_memory(video_url, text)

            st.success("✅ Video processed!")

            st.text_area(
                "📄 Transcript Preview",
                st.session_state.transcript[:2000],
                height=200
            )


# ---------------- Q&A SECTION ---------------- #

if "retriever" in st.session_state:

    st.subheader("💬 Ask a question about the video")

    question = st.text_input("Your question", key="question_input")

    if st.button("🤖 Get Answer", key="get_answer"):

        if not question:
            st.warning("Please enter a question.")

        else:

            retriever = st.session_state.retriever

            docs = retriever.invoke(question)

            context_text = "\n\n".join(doc.page_content for doc in docs)

            context_text = context_text[:6000]

            prompt = PromptTemplate(
                template="""
You are an expert assistant.

Using the transcript context, answer the question in a short paragraph (5–7 sentences).
Explain the concept clearly and briefly.

Do not give one-line answers.

Context:
{context}

Question:
{question}

Detailed Answer:
""",
                input_variables=["context", "question"]
            )

            final_prompt = prompt.invoke({
                "context": context_text,
                "question": question
            })

           

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.3,
                api_key=st.secrets["GROQ_API_KEY"]
            )

            with st.spinner("Thinking..."):

                answer = llm.invoke(final_prompt)

            st.success("✅ Answer")
            st.write(answer.content)


