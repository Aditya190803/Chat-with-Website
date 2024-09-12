import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

@st.cache_data
def get_embedding_model(model_name):
    return HuggingFaceBgeEmbeddings(model_name=model_name)

@st.cache_data
def get_vectorstore_from_url(url, _embedding_model):  # Note the leading underscore
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, _embedding_model)

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGroq(temperature=0.35, groq_api_key=st.secrets["GROQ_API_KEY"], model_name="mixtral-8x7b-32768")
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model_name="mixtral-8x7b-32768")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
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

# app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    with st.form(key='url_form'):
        website_url = st.text_input("Website URL")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        st.session_state.website_url = website_url

if "website_url" in st.session_state:
    website_url = st.session_state.website_url
    if website_url:
        # session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello, I am a bot. How can I help you?")
            ]
        if "vector_store" not in st.session_state:
            embedding_model = get_embedding_model("BAAI/bge-small-en")
            st.session_state.vector_store = get_vectorstore_from_url(website_url, embedding_model)    

        # user input
        user_query = st.chat_input("Type your message here...")
        if user_query:
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            
        # conversation
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
else:
    st.info("Please enter a website URL and press Submit.")
