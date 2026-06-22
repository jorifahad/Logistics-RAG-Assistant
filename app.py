import streamlit as st # UI framework for web app
import time            # used for typing effect animation
from get_embedding_function import get_embedding_function   # custom embedding function
from langchain_community.vectorstores import Chroma                 # vector database (stores embeddings)
from query_data import query_rag

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Logistics RAG Assistant",       # title of tab
    page_icon="logos\icon.png",                   # icon
    layout="wide",                          # full width layout
)

# 2. CUSTOM CSS (STYLING)
st.markdown("""
<style>

:root {
    --bg: #f5efe9;
    --primary: #f5efe9;
    --secondary: #996951;
    --card: #e6c9b8;
    --text: #7b5542;
}

/* Main App */
.stApp {
    background-color:;
    color: var(--text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--primary);
    color: white;
}

/* Titles */
.title {
    font-size: 40px;
    font-weight: 800;
    color: var(--text);
}

.text {
    font-size: 32px;
    font-weight: 600;
    color: var(--text);
}

.text2 {
    font-size: 18px;
    font-weight: 400;
    color: var(--text);
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    background-color: var(--primary);
    border-radius: 12px;
    padding: 10px;
    color: #996951;
}

/* Chunk box */
.chunk-box {
    background-color: var(--card);
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid var(--secondary);
}


/* Clear Chat Button */
div.stButton > button {
    background-color: #996951;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1em;
    font-weight: 600;
}

/* Source box */
.source-box {
    background-color: var(--card);
    padding: 10px;
    border-radius: 10px;
    border-left: 4px solid var(--secondary);
}

/* Input box */
.stChatInputContainer {
    border-top: 2px solid var(--secondary);    
}

</style>
""", unsafe_allow_html=True)
CHROMA_PATH = "chroma"

@st.cache_resource
def load_vectorstore():

    embedding_function = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    return db
# 3. SIDEBAR (CONTROLS)
with st.sidebar:

    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logos\setiings.png", width=50)
    with col2:
        st.markdown('<div class="text">Settings',
                    unsafe_allow_html=True
        )
    st.divider()
    st.markdown('<div class="text2">Number of retrieved results',
                    unsafe_allow_html=True)
    top_k = st.slider("Number of retrieved results", min_value=1, max_value=10, value=3, label_visibility="collapsed")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []


# 4. HEADER SECTION
col1, col2 = st.columns([1, 20])
with col1:
    st.image("logos\icon.png", width=60)

with col2:
    st.markdown('<div class="title">Logistics Assistant</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle"> An AI-powered Retrieval-Augmented Generation system designed to assist logistics operations <br> through intelligent document understanding and context-aware responses.</div>',
    unsafe_allow_html=True
)
st.write("")

# 5. SESSION STATE (CHAT HISTORY)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. CHAT INPUT & RAG EXECUTION
prompt = st.chat_input("Ask a logistics question...")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user","content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute RAG Pipeline
    with st.spinner("Analyzing documents..."):
        db = load_vectorstore()
        # Calling the function from rag_engine.py
        response, retrieved_chunks, sources = query_rag(prompt, top_k,db)

        # Display Assistant Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for word in response.split():
                full_response += word + " "
                time.sleep(0.02)
                placeholder.markdown(full_response)

        # Save assistant response
        st.session_state.messages.append({"role": "assistant","content": response})

        # # 7. RETRIEVED CONTEXT SECTION
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        # LEFT COLUMN
        with col1:
            header_col1, header_col2 = st.columns([1, 15])
            with header_col1:
                st.image("logos\\retrieval.png", width=50)
            with header_col2:
                st.markdown("### Retrieved Chunks")
                
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"""
                <div class="chunk-box">
                    <b style="color: var(--secondary);">Chunk #{i+1}</b><br><br>
                    {chunk}
                </div>
                """, unsafe_allow_html=True)
                st.write("") 

        # RIGHT COLUMN
        with col2:
            header_col1, header_col2 = st.columns([1, 7.5])
            with header_col1:
                st.image("logos\source.png", width=50)
            with header_col2:
                st.markdown("### Sources")
                
            for source in sources:
                st.markdown(f"""
                <div class="source-box">
                    {source}
                    </div>
                """, unsafe_allow_html=True)
                st.write("") 
