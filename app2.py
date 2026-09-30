import streamlit as st
import ollama

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 My AI Chatbot")
st.caption("Powered by Ollama + Streamlit")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Get response from Ollama
    response = ollama.chat(
        model="llama3.2",
        messages=st.session_state.messages
    )

    bot_response = response["message"]["content"]

    # Display bot response
    with st.chat_message("assistant"):
        st.write(bot_response)

    # Store bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })