import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os

# TITLE
st.title("📚 Research Paper Q&A Assistant")

# PDF TEXT
def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            text += page.extract_text()

    return text

# CHUNKS
def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks

# VECTOR STORE
def get_vector_store(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")

# QA CHAIN
def get_chain():

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the provided context.

        Context:
        {context}

        Question:
        {input}
        """
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
    )

    chain = create_stuff_documents_chain(
        model,
        prompt
    )

    return chain

# QUESTION
def answer_question(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question)

    chain = get_chain()

   response = chain.invoke(
    {
        "context": docs,
        "input": user_question
    }
)

    st.write("## Answer")
   st.write(response)

# SIDEBAR
with st.sidebar:

    st.header("Upload PDFs")

    pdf_docs = st.file_uploader(
        "Upload Research Papers",
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):

        with st.spinner("Processing..."):

            raw_text = get_pdf_text(pdf_docs)

            chunks = get_text_chunks(raw_text)

            get_vector_store(chunks)

            st.success("Done")

# QUESTION INPUT
user_question = st.text_input("Ask Question")

if user_question:

    answer_question(user_question)
