import streamlit as st
import os
from openai import OpenAI
from google import genai
import ollama
import time
from dotenv import load_dotenv

load_dotenv()

# ===============Page Configuration===================
st.set_page_config(
    page_title = "Multi-Model AI App",
    layout = "wide"
)

st.title("Multi-Model AI App")
st.caption("Compare and Interact with different kind of LLMS....")

# ===========================Sidebar======================
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("Select the Provider",
                            ["OpenAi",
                             "Google Gemini",
                             "Ollama",
                             "Compare All"
                            ])

    st.divider()

    temperature = st.slider(
        "Temperature",
        min_value = 0.0,
        max_value = 2.0,
        value = 0.7,
        step = 0.1
    )

    max_tokens = st.slider(
        "Maximum Output Tokens",
        min_value = 100,
        max_value = 3000,
        value = 1000,
        step = 100
    )
    st.divider()

    system_prompt = st.text_area(
        "System Prompt",
        value = "You are a helpful AI assistant"
    )

    st.divider()

    if st.button("Clear Chat",use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========MODELS=================
OPENAI_MODEL = "gpt-4.1-mini"
GEMINI_MODEL = "gemini-3-flash-preview"
OLLAMA_MODEL = "gemma3"

# ===========Initializing the Models====================
def get_openai_provider():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None
    return OpenAI(api_key=api_key)

def get_google_gemini_provider():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None
    return genai.Client(api_key=api_key)


# =============OPENAI=============================
def ask_openai(prompt, system_prompt, temperature, max_tokens):

    try:
        client = get_openai_provider()
        if client is None:
            return "OPENAI_API_KEY is not configured...!"

        response = client.responses.create(
            model = OPENAI_MODEL,
            instructions = system_prompt,
            input = prompt,
            temperature = temperature,
            max_output_tokens = max_tokens
        )

        return response.output_text

    except Exception as e:
        return f"OpenAi Error: {str(e)}"


# ============GEMINI=====================
def ask_gemini(prompt, system_prompt, temperature, max_tokens):
    client = get_google_gemini_provider()

    if not client:
        return "GEMINI_API_KEY is not configured....!"

    response = client.models.generate_content(
        model = GEMINI_MODEL,
        contents = prompt,
        config = {
            "system_instruction":system_prompt,
            "temperature":temperature,
            "max_output_tokens":max_tokens
        }
    )

    return response.text



# ============OLLAMA=====================
def ask_ollama(prompt, system_prompt, temperature, max_tokens):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )
        return response["message"]["content"]

    except Exception as e:
        return f"Ollama Error: {str(e)}"


# ============MODEL ROUTER==================
def generate_response(
        provider,
        prompt,
        system_prompt,
        temperature,
        max_tokens
    ):

    start_time = time.time()

    try:
        if provider == "OpenAi":
            response = ask_openai(
                prompt,
                system_prompt, 
                temperature, 
                max_tokens
            )
        elif provider == "Google Gemini":
            response = ask_gemini(
                prompt,
                system_prompt,
                temperature,
                max_tokens
            )

        elif provider == "Ollama":
            response = ask_ollama(
                prompt,
                system_prompt,
                temperature,
                max_tokens
            )

        else:
            response = "Invalid Provider"

        execution_time = round(time.time() - start_time,2)

        return response, execution_time
        
    except Exception as e:
        execution_time = round(
            time.time() - start_time,
            2
        )

        return f"{provider} Error: {str(e)}", execution_time

# ==========Session State=====================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ========Displaing Chat History===================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ========User Input===============
prompt = st.chat_input(
    "Ask something...."
)

# =======PROCESS PROMPT======
if prompt:

    # Storing User Message
    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
         }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # ========COMPARE ALL MODELS=============
    if provider == "Compare All":
        st.subheader("Model Comparison")

        models = [
            "OpenAi",
            "Google Gemini",
            "Ollama"
        ]

        results = {}

        for model in models:

            with st.spinner(
                f"Generating response from {model}"
            ):
                response, execution_time = generate_response(
                    model,
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens
                )

                results[model] = {
                    "response" : response,
                    "time" : execution_time
                }

        # Display results
        columns = st.columns(3)

        for index, model in enumerate(models):
            with columns[index % 2]:
                st.markdown(
                    f"### {model}"
                )
                st.write(
                    results[model]["response"]
                )
                st.caption(
                    f"Response Time: "
                    f"{results[model]['time']} seconds"
                )

    else:
        with st.chat_message("assistant"):
            with st.spinner(
                f"{provider} is thinking...."
            ):
                response, execution_time = generate_response(
                    provider,
                    prompt,
                    system_prompt,
                    temperature,
                    max_tokens
                    )
            st.markdown(response)
            st.caption(
                f"Model: {provider} |"
                f"{execution_time} seconds |"
                f"Temperature: {temperature}"
            )

        # Save the assistant Response
        st.session_state.messages.append({
            "role": "assistant",
            "content":response
        })


    

