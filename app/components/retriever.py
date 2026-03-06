# app/components/retriever.py

from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """You are a medical assistant. Answer the following question in 2-3 lines maximum using ONLY the information provided in the context. If the answer is not present in the context, say "I don't know" briefly and do not speculate.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

def _format_docs(docs):
    # Combine retrieved documents into a context string
    return "\n\n".join(d.page_content for d in docs)

def create_qa_chain():
    """
    Returns a Runnable that expects {"input": "<user question>"} and
    outputs a plain answer string.
    """
    try:
        logger.info("Loading vector store for retrieval...")
        db = load_vector_store()
        if db is None:
            raise CustomException("Vector store not present or empty")

        retriever = db.as_retriever(search_kwargs={"k": 2})  # tweak k if needed

        llm = load_llm()
        if llm is None:
            raise CustomException("LLM not loaded")

        prompt = set_custom_prompt()

        # Build an LCEL pipeline:
        # 1) Take {"input": question}
        # 2) Create {"context": retrieved_text, "question": question}
        # 3) Prompt -> LLM -> string
        rag_chain = (
            {
                "context": itemgetter("input") | retriever | RunnableLambda(_format_docs),
                "question": itemgetter("input"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Successfully created the QA chain (Groq + FAISS) using LCEL primitives.")
        return rag_chain

    except Exception as e:
        error_message = CustomException("Failed to create QA chain", e)
        logger.error(str(error_message))
        raise error_message