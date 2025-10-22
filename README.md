
# Multi-Agent Airline Conversational AI

This project is a sophisticated, multi-agent conversational AI system designed to provide natural, context-aware, and policy-compliant customer support for the airline industry.

## Table of Contents

* [Problem Statement](#problem-statement)
* [Our Solution](#our-solution)
* [Workflow](#workflow)
* [Tech Stack](#tech-stack)
* [Repository Structure](#repository-structure)
* [Setup & Running the Demo](#setup--running-the-demo)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Environment Variables](#environment-variables)
  * [Running the Backend Services](#running-the-backend-services)
  * [Running the Streamlit Demo](#running-the-streamlit-demo)
* [Team & Contributions](#team--contributions)

## Problem Statement

Standard customer support chatbots often feel robotic and frustrating. They struggle with multi-turn conversations, fail to understand context, and cannot adhere to complex company policies. This leads to a poor user experience and requires frequent escalation to human agents. The challenge is to build a bot that can understand conversational flow, retain critical information, and provide responses that are not only accurate but also empathetic and aligned with the airline's operational guidelines.

## Our Solution

We solve this problem by implementing a **multi-agent microservices architecture**. Instead of a single monolithic model, our system delegates tasks to specialized agents, each responsible for one part of the conversational pipeline.

* **Intent Routing**: A fine-tuned classifier first understands what the user wants.
* **Context Management**: A dedicated agent maintains a structured summary of the conversation, acting as the bot's short-term memory.
* **Policy Retrieval (RAG)**: An agent retrieves relevant policy snippets from a knowledge base to ensure all responses are compliant and accurate.
* **Response Generation**: A final agent synthesizes all this information to generate a natural, helpful, and policy-adherent response.

This modular approach makes the system more robust, scalable, and easier to debug, directly addressing the core issues of context retention and policy compliance.

## Workflow

The system follows a sequential pipeline where the output of one component becomes the input for the next. The `demo.py` application orchestrates the calls to each microservice.

<img width="1131" height="2003" alt="flow" src="https://github.com/user-attachments/assets/22aad90b-c518-4c6a-9f35-703ff76e5c37" />


1. **Intent Routing**: The user's query is passed to the intent classifier to determine the intent.
2. **Context Management**: A dedicated agent maintains the conversation history and updates the context summary.
3. **Policy Retrieval (RAG)**: The policy retrieval agent ensures that the response adheres to airline policies by fetching relevant snippets.
4. **Response Generation**: The final agent generates a natural response, ensuring it aligns with both the context and the airline's operational guidelines.

## Tech Stack

* **Backend & APIs**: Python, FastAPI
* **Frontend & Demo**: Streamlit
* **ML / NLP**:

  * **Intent Classification**: Hugging Face Transformers, Scikit-learn
  * **Context & Response Generation**: Large Language Models (e.g., Google Gemini)
  * **RAG**: Sentence-Transformers, Vector Databases (e.g., ChromaDB)
* **Data Handling**: Pandas, NumPy

## Repository Structure

```
.
├── context_management.py         # Handles conversation context
├── intent_classify.py            # Intent classification agent
├── fine_tune_classifier.py       # Fine-tuned classifier training
├── Policy_retrival_RAG.ipynb     # Policy retrieval with RAG
├── response_generation.py        # Response generation logic
├── demo.py                       # Main orchestration of the system
├── requirements.txt              # Required dependencies
└── .env                          # Store your API keys here
```

## Setup & Running the Demo

### Prerequisites

* Python 3.9+
* An LLM API Key (e.g., from Google AI Studio)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/airline-conversational-ai.git
   cd airline-conversational-ai
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

1. Create a `.env` file in the root directory to store your API key:

   ```
   LLM_API_KEY=your_api_key_here
   ```

### Running the Backend Services

You need to run each of the four backend services in separate terminals.

1. **Terminal 1**: Run the **Intent Classifier** service:

   ```bash
   uvicorn intent_classify:app --reload
   ```

2. **Terminal 2**: Run the **Context Management** service:

   ```bash
   uvicorn context_management:app --reload
   ```

3. **Terminal 3**: Run the **Policy Retrieval (RAG)** service:

   ```bash
   uvicorn Policy_retrival_RAG:app --reload
   ```

4. **Terminal 4**: Run the **Response Generation** service:

   ```bash
   uvicorn response_generation:app --reload
   ```

### Running the Streamlit Demo

Once all backend services are running, open a fifth terminal and run the frontend:

```bash
streamlit run demo.py
```

Navigate to `http://localhost:8501` in your browser to interact with the chatbot.

## Team & Contributions

This project was a collaborative effort by:

* **Surya**: Context Management (`context_management.py`)
* **Swadhi**: Intent Routing (`intent_classify.py`, `fine_tune_classifier.py`)
* **Aswath**: Policy Retrieval & RAG (`Policy_retrival_RAG.ipynb`)
* **Mukhil**: Core Logic & Response Generation (`response_generation.py`)
* **Group Effort**: Final Integration & Demo (`demo.py`)

