import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

<<<<<<< HEAD
open_api_key = os.getenv("GOOGLE_API_KEY")

def maha():
    
    raddi = st.sidebar.radio("Chat With P.A", ["Text Chat", "Doc Chat"])
    with st.sidebar:
        temprature= st.slider("How much creative you want",0.0,1.0,0.1)
    if raddi == "Text Chat":
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])
=======
import streamlit as st
import os
import google.generativeai as genai 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])
>>>>>>> 3969112951139e885af9553d51eedd42d20efff2

        def get_gemini_response(question):
            response = chat.send_message(question, stream=True)
            return response

        # st.set_page_config(page_title="Q&A gemini")
        st.header("Prajwals Pa")

        if 'Chat_history' not in st.session_state:
            st.session_state['Chat_history'] = []

        input_text = st.text_area("Input:", key=101)
        submit = st.button("Get Your Answer")

<<<<<<< HEAD
        if submit and input_text:
            response = get_gemini_response(input_text)
            st.session_state["Chat_history"].append(("You", input_text))
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)
                st.session_state["Chat_history"].append(("Bot", chunk.text))

        st.subheader("Chat History")
        for role, text in st.session_state["Chat_history"]:
            st.write(f"{role}: {text}")

    elif raddi == "Doc Chat":
        def get_pdf_text(pdf_docs):
            text = ""
            for pdf in pdf_docs:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    text += page.extract_text()  
            return text

        def get_text_chunks(text):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            chunks = text_splitter.split_text(text)
            return chunks

        def get_vector_store(text_chunks):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            if text_chunks:
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                vector_store.save_local("faiss_index")
            else:
                print("Text chunks list is empty.")

        def get_conversational_chain():
            prompt_template = """
            Answer the question as detailed as possible 
            Context : \n {context}?\n
            Question: \n {question}?\n

            Answer:
            """
            model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temprature)  
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
            return chain

        def user_input(user_question):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain(
                {
                    "input_documents": docs, "question": user_question
                },
                return_only_outputs=True
            )
            st.write("Reply: ", response["output_text"])

        # st.set_page_config("Chat Pdf")
        st.header("Chat with the pdfs")
        user_question = st.text_input("Ask a question from Pdf files")
        if user_question:
            user_input(user_question)
        with st.sidebar:
            
            st.title("Chat with pdfs")
            pdf_docs = st.file_uploader("Upload Your Pdf files and click submit", type="pdf", accept_multiple_files=True)
            if st.button("submit & process"):
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    maha()
=======
if submit and input_text:
    response = get_gemini_response(input_text)
    st.session_state["Chat_history"].append(("You", input_text))
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)

        st.session_state["Chat_history"].append(("Bot", chunk.text))

st.subheader("Chat History")
for role, text in st.session_state["Chat_history"]:
    st.write(f"{role}: {text}")
>>>>>>> 3969112951139e885af9553d51eedd42d20efff2
