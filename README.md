# Chat with Websites 

This project is a **Chatbot** that lets users interact with the content of websites by embedding them into a **retriever-augmented generation (RAG) pipeline**. The app extracts content from a website, uses **Hugging Face embeddings** for vectorization, and **ChatGroq LLM** to generate conversational responses.

The chatbot supports **context-aware responses** based on previous interactions, making it a useful tool for obtaining targeted information directly from websites.

---

## Features

- **Embed Website Content:** Extracts content from any URL for querying.
- **Conversation Awareness:** Keeps track of previous interactions to provide meaningful answers.
- **Retriever-Augmented Generation (RAG):** Combines content retrieval with generative responses.
- **Interactive UI:** User-friendly chat interface powered by **Streamlit**.
- **State Management:** Maintains chat history and embedded content using **Streamlit session state**.

---

## Prerequisites

Before running the project, make sure you have:

1. **Python 3.8+** installed.
2. **API key for ChatGroq LLM:**  
   - Sign up and generate your key from the **[Groq API portal](https://console.groq.com/keys)**.

---

## How to Clone and Run the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Aditya190803/chat-with-website.git
   cd chat-with-website
   ```

2. **Set up API key:**

   - Create a `.streamlit/secrets.toml` file in the root directory with the following content:
     ```
     GROQ_API_KEY=your_groq_api_key
     ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

5. **Access the app:**  
   Open your browser and navigate to `http://localhost:8501`.

---

## Usage

1. **Enter the website URL** in the sidebar and click **Submit**.
2. Once the content is embedded, you can **type questions** in the chat interface.
3. The chatbot will **respond based on the website content** and previous conversation history.
4. **Continue interacting** naturally, with responses becoming more relevant as the conversation progresses.
