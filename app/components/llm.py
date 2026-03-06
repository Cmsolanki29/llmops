from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY, GROQ_MODEL

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(groq_api_key: str = GROQ_API_KEY, groq_model: str = GROQ_MODEL):
    """
    Load Groq Chat model as a LangChain ChatModel.
    """
    try:
        logger.info(f"Loading LLM from Groq | model={groq_model}")
        if not groq_api_key:
            raise CustomException("GROQ_API_KEY is missing. Set it in your environment/.env")

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=groq_model,
            temperature=0.3,
            max_tokens=256,
        )

        logger.info("LLM (Groq) loaded successfully.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load Groq LLM", e)
        logger.error(str(error_message))
        raise error_message