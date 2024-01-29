from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai 
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

st.set_page_config(page_title="Q&A gemini")
st.header("Prajwals Pa")

if 'Chat_history' not in st.session_state:
    st.session_state['Chat_history'] = []

input_text = st.text_area("Input:",key=101)
submit = st.button("Get Your Answer")

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
