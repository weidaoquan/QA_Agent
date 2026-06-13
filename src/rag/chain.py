"""RAG chain assembly: retrieval-augmented generation pipeline."""

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.memory.history import create_memory_manager


# System prompt that instructs the LLM how to use the retrieved context
QA_SYSTEM_PROMPT = """\
You are a helpful company internal knowledge assistant. \
Your role is to answer questions based ONLY on the provided context documents.

Instructions:
- Answer the question using the context provided below.
- If the context contains the answer, provide it clearly and concisely.
- If the answer cannot be found in the context, say \
"I could not find information about that in the knowledge base. \
Try rephrasing your question or adding relevant documents."
- Do NOT make up or infer information beyond what is in the context.
- Cite the source document when possible.

Context:
{context}"""

# Prompt that generates a standalone question from chat history
CONTEXTUALIZE_Q_SYSTEM_PROMPT = """\
Given the chat history and the latest user question, \
formulate a standalone question that can be understood without the chat history. \
Do NOT answer the question, just reformulate it if needed. \
If the question is already clear and standalone, return it as is."""


def create_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
) -> Runnable:
    """Assemble the full RAG chain with history-aware retrieval.

    The chain flow:
    1. History-aware question rephrasing → standalone question
    2. Standalone question → retriever → relevant documents
    3. Documents + question → LLM (via stuff chain) → answer

    Args:
        retriever: Document retriever (e.g., from Chroma vector store).
        llm: Chat model for generation.

    Returns:
        A Runnable chain that accepts {"input": "user question"} and
        returns {"answer": "...", "context": [...]}.
    """
    # Prompt for rephrasing the question using chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # History-aware retriever: rephrases question before retrieval
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Prompt for the final QA step
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Stuff documents chain: formats context into the prompt and generates
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Full retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def create_conversational_rag_chain(
    retriever: BaseRetriever,
    llm: BaseChatModel,
    memory_file: str = "conversation_history.json",
) -> RunnableWithMessageHistory:
    """Create a conversational RAG chain with persistent chat history.

    Wraps the RAG chain with RunnableWithMessageHistory for automatic
    chat history tracking across conversation turns.

    Args:
        retriever: Document retriever.
        llm: Chat model for generation.
        memory_file: Path to the JSON file for persisting chat history.

    Returns:
        A RunnableWithMessageHistory that manages chat history automatically.
    """
    rag_chain = create_rag_chain(retriever, llm)
    memory_manager = create_memory_manager(memory_file)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        memory_manager.get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain
