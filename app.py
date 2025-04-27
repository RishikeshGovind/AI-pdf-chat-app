import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from huggingface_hub.utils import HfHubHTTPError
from htmlTemplates import css, bot_template, user_template

# ========== Helper Functions ==========

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def safe_generate_response(conversation, user_question):
    try:
        response = conversation({'question': user_question})
        return response
    except HfHubHTTPError:
        st.error("🚨 Hugging Face server is busy. Please try again in a few minutes!")
        return None

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"), 
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    with st.spinner("🤖 Generating answer... Please wait..."):
        response = safe_generate_response(st.session_state.conversation, user_question)

    if response is None:
        return  # Error already shown, skip rendering

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# ========== Main App ==========

def main():
    st.set_page_config(
        page_title="Chat with your PDFs 📚",
        page_icon=":books:"
    )

    load_dotenv()

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your PDFs 📚")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        st.info("⚡ Tip: Upload smaller PDFs (<10MB) for best performance!")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your PDFs... ⏳"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("✅ PDFs processed successfully! Ask me anything about them.")
            else:
                st.warning("⚠️ Please upload at least one PDF before processing!")

# ========== Run the App ==========

if __name__ == '__main__':
    main()
