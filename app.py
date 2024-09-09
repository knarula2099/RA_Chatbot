import streamlit as st
import os
from local_search import create_search_engine


def app():
    api_key = st.secrets['GRAPHRAG_API_KEY']
    st.title('RA Handbook Chatbot')

    search_engine = create_search_engine(api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

    if prompt := st.chat_input("Type something..."):
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('bot'):
            response = response_generator(prompt, search_engine)
            st.markdown(response)

        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })
        st.session_state.messages.append({
            'role': 'bot',
            'content': response
        })


def response_generator(prompt, search_engine):
    response = search_engine.search(prompt)
    return response.response


app()
