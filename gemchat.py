from Crypto.Cipher import AES
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from importlib.metadata import distribution, metadata, version
from PIL import Image
# from langchain_google_genai import ChatGoogleGenerativeAI
# from pdfplumber import PdfReader
# from haystack.preprocessor.utils import RecursiveCharacterTextSplitter
# from haystack.document_store.faiss import FAISSDocumentStore
# from haystack.retriever.dense import DensePassageRetriever
# from haystack.pipeline import ExtractiveQAPipeline
import os

load_dotenv()

#api_key = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = "AIzaSyAcQsUWRPmvHEUrSZ3IagetZFbgns49vto"
genai.configure(api_key=GOOGLE_API_KEY)
def maha():
    
    raddi = st.sidebar.radio("Chat With P.A", ["Text Chat", "Doc Chat","Image Chat"])
    # with st.sidebar:
    #     temperature = st.slider("How much creative you want", 0.0, 1.0, 0.1)

    if raddi == "Doc Chat":
        with st.sidebar:
            temperature = st.slider("How much creative you want", 0.0, 1.0, 0.1)
    
        def get_pdf_text(pdf_docs):
           
           text=""
           for pdf in pdf_docs:
               
               
               pdf_reader= PdfReader(pdf)
               for page in pdf_reader.pages:
                   
                   text+= page.extract_text()
               return  text
    
    
    
        def get_text_chunks(text):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            chunks = text_splitter.split_text(text)
            return chunks
    
    
        def get_vector_store(text_chunks):
            embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
    
    
        def get_conversational_chain():
        
            prompt_template = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
        
            model = ChatGoogleGenerativeAI(model="gemini-pro",
                                     temperature=0.3)
        
            prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        
            return chain
    
    
    
        def user_input(user_question):
            embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
            
            new_db = FAISS.load_local("faiss_index", embeddings)
            docs = new_db.similarity_search(user_question)
        
            chain = get_conversational_chain()
        
            
            response = chain(
                {"input_documents":docs, "question": user_question}
                , return_only_outputs=True)
        
            print(response)
            st.write("Reply: ", response["output_text"])

    
        st.header("Chat with PDF using Gemini💁")
        
        user_question = st.text_input("Ask a Question from the PDF Files")
    
        if user_question:
            user_input(user_question)
    
        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
        
        
    elif raddi == "Text Chat":
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat(history=[])
    
        def get_gemini_response(question):
            response = chat.send_message(question, stream=True)
            return response
    
        st.header("Prajwals Pa")
    
        if 'Chat_history' not in st.session_state:
            st.session_state['Chat_history'] = []
    
        input_text = st.text_area("Input:", key=101)
        submit = st.button("Get Your Answer",key = 103)
    
        if submit and input_text:
            response = get_gemini_response(input_text)
            st.session_state["Chat_history"].append(("You", input_text))
            st.subheader("The Response is")
            for chunk in response:
                st.markdown(chunk.text)
                st.session_state["Chat_history"].append(("Bot", chunk.text))
    
        st.subheader("Chat History")
        for role, text in st.session_state["Chat_history"]:
            st.write(f"{role}: {text}")
    
    elif raddi == "Image Chat":
        # Function to prepare image data
        def prepare_image_data(uploaded_file):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                return image
            else:
                raise FileNotFoundError("No file uploaded")
    
        # Function to get Gemini response using image data
        def get_gemini_response(input_prompt, image_data):
            model = genai.GenerativeModel('gemini-pro-vision')
            response = model.generate_content([input_prompt, image_data])
            return response
    
        st.header("Chat with the Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is None:
            uploaded_file = st.camera_input("Take Photo")
    
        if uploaded_file is not None:
            image = prepare_image_data(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
            submit = st.button("Tell me about the Image")
            input_text = st.text_input("Type Your Query About The Image", key=101)
            input_prompt = f"""
            You are an expert in all domains. Please explain detailed information about the image: {input_text}
            """
            if submit:
                response = get_gemini_response(input_prompt, image)
                st.header("The Response is")
                st.markdown(response.text)



if __name__ == "__main__":
    maha()
