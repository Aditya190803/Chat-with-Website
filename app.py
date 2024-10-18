import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def get_vectorstore_from_url(url):
    # Get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # Create a vectorstore from the chunks using Hugging Face embeddings
    vector_store = Chroma.from_documents(document_chunks, HuggingFaceEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq()  # Use Groq as the LLM
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain):
    llm = ChatGroq()  # Use Groq as the LLM
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# Main title
st.title("Chat with Websites")

# Initialize session state variables if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I assist you today?"),
    ]

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # Placeholder, will be set after URL input

# Sidebar contents
with st.sidebar:
    website_url = st.text_input("Website URL", placeholder="Enter the URL of the website")
    submit_button = st.button(label="Submit")

    if submit_button:
        st.session_state.website_url = website_url

    if website_url and not st.session_state.vector_store:
        with st.spinner("Embedding the website..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
            st.success("Website successfully embedded. You can now ask questions.")

# User input
if st.session_state.vector_store:
    user_query = st.chat_input("Type your message here...")

    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
else:
    user_query = None  
    
# Display conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
