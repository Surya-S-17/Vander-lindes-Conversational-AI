import streamlit as st
import requests

CONTEXT_SUMMARY_API = "http://1227.0.0.1:8001/summarize_context"
INTENT_ROUTER_API = "http://127.0.0.1:8000/predict-intent/"
RAG_API = "http://127.0.0.1:8002/get-policies/"
RESPONSE_GENERATION_API = "http://127.0.0.1:8003/generate-response/"

def api_post_request(url: str, json_data: dict, error_message: str) -> dict:
    try:
        response = requests.post(url, json=json_data)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
        st.error(f"API Error ({error_message}): {e}")
        return {"error": str(e)}

def classify_intent(user_query: str) -> str:
    response = api_post_request(
        url=INTENT_ROUTER_API,
        json_data={"query": user_query},
        error_message="Intent Router"
    )
    return response.get("intent", "api_error")

def retrieve_rag_content(intent: str) -> list:
    response = api_post_request(
        url=RAG_API,
        json_data={"intent": intent},
        error_message="RAG/Policy Service"
    )
    return response.get("policies", ["RAG API connection failed."])

def generate_context_summary(history: list, new_message: str) -> str:
    response = api_post_request(
        url=CONTEXT_SUMMARY_API,
        json_data={"history": history, "new_message": new_message},
        error_message="Context Summary"
    )
    return response.get("context_summary", "Context API connection failed.")

def generate_final_response(intent: str, context: str, rag_snippets: list) -> str:
    response = api_post_request(
        url=RESPONSE_GENERATION_API,
        json_data={"intent": intent, "context": context, "rag_snippets": rag_snippets},
        error_message="Response Generation"
    )
    return response.get("response", "Response generation failed.")

def main():
    st.set_page_config(page_title="Airline Chatbot", layout="wide")
    st.title("✈️ Airline Chatbot Demo")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.latest_intent = {}
        st.session_state.latest_context = ""
        st.session_state.latest_rag_response = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you today?"})

    with st.sidebar:
        st.header("⚙️ Inner Workings")
        st.markdown("This panel shows the real-time analysis for the latest user message.")
        
        st.subheader("1. Detected Intent", divider="blue")
        st.json(st.session_state.latest_intent)

        st.subheader("2. Retrieved Policies (RAG)", divider="blue")
        st.text_area("Policies", "\n".join(st.session_state.latest_rag_response), height=150)
        
        st.subheader("3. Generated Context", divider="blue")
        st.text_area("Summary", st.session_state.latest_context, height=150)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about booking a flight..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                intent = classify_intent(prompt)
                st.session_state.latest_intent = {"intent": intent}
                
                rag_response = retrieve_rag_content(intent)
                st.session_state.latest_rag_response = rag_response
                
                history_for_api = [msg for msg in st.session_state.messages if msg["role"] != "assistant"]
                context_summary = generate_context_summary(history=history_for_api, new_message=prompt)
                st.session_state.latest_context = context_summary

                bot_response = generate_final_response(intent, context_summary, rag_response)

                st.markdown(bot_response)

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        st.rerun()

if __name__ == "__main__":
    main()
