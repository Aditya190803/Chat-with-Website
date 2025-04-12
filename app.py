import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

# ✅ Fix watcher warning (Streamlit + torch)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ✅ Optional: Add a user agent for better request identification
os.environ["USER_AGENT"] = "ChatWithWebsiteBot/1.0"

# ✅ Specify Groq model
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ✅ Cache the vector store to avoid recomputation
@st.cache_resource(show_spinner=False)
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # Create embeddings and store vectors in FAISS
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(document_chunks, embedding=embeddings)
    return vector_store

# ✅ Create a history-aware retriever chain
def get_context_retriever_chain(vector_store):
    llm = ChatGroq(model=GROQ_MODEL)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

# ✅ Create a retrieval-augmented generation (RAG) chain
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatGroq(model=GROQ_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# ✅ Generate a response based on user input
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# ✅ Streamlit App UI
st.set_page_config(page_title="Chat with Website", page_icon="🤖")
st.title("🤖 Chat with Website")

# Sidebar
with st.sidebar:
    st.header("Settings")
    if "website_url" not in st.session_state:
        st.session_state.website_url = ""
    
    website_url = st.text_input("Enter Website URL", value=st.session_state.website_url)
    submit_button = st.button("Submit")

# Handle new website submission
if submit_button:
    st.session_state.website_url = website_url
    if website_url:
        with st.spinner("🔄 Loading and processing website content..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
        st.success("✅ Website content loaded! Start chatting below.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?")
    ]

# Chat section
if "vector_store" in st.session_state:
    user_query = st.chat_input("Type your message here...")
    if user_query:
        try:
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            st.error(f"💥 An error occurred: {e}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
            st.write(message.content)
else:
    st.info("ℹ️ Please enter a website URL and click Submit.")
