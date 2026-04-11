from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

def create_rag_chain(retriever):
    # ✅ LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0   # important for RAG
    )

    # ✅ 1. History-Aware Reformulation Pipeline
    # This prompt rephrases the user's question to be standalone, based on chat history.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_maker = contextualize_q_prompt | llm | StrOutputParser()

    # Dynamic retrieval function
    def retrieve_context(inputs):
        chat_history = inputs.get("chat_history", [])
        # Only rephrase if there is actually chat history
        if not chat_history:
            standalone_q = inputs["input"]
        else:
            standalone_q = question_maker.invoke(inputs)
            
        docs = retriever.invoke(standalone_q)
        return "\n".join([doc.page_content for doc in docs])

    # ✅ 2. Answer Chain
    qa_system_prompt = (
        "You are a helpful assistant.\n"
        "You have been given a youtube video, act like a teacher.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not in the context, say: 'I don't know. Please contact support.'\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Return dict format that matches Streamlit app expectations
    def format_output(response_str):
        return {"answer": response_str}

    # ✅ 3. Final Standard LCEL Chain
    rag_chain = (
        RunnablePassthrough.assign(context=retrieve_context)
        | qa_prompt | llm  | StrOutputParser()  | format_output
    )

    # rag_chain.get_graph().print_ascii()

    return rag_chain