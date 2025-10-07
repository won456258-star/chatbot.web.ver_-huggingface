# 파일명: rag_logic.py

import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# web.py에서 에러를 잡아 처리할 수 있도록 사용자 정의 에러를 만듭니다.
class RagChainInitializationError(Exception):
    pass

@st.cache_resource
def get_rag_chain(api_key: str):
    """
    Hugging Face API 키를 인자로 받아 RAG 체인을 생성하고 반환합니다.
    오류 발생 시 RagChainInitializationError를 발생시킵니다.
    """
    try:
        # 1. 임베딩 모델 설정
        embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 2. FAISS 벡터 저장소 로드
        vectorstore = FAISS.load_local(
            "my_faiss_db",
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever()

        # 3. LLM 초기화 (전달받은 API 키 사용)
        HUGGING_FACE_MODEL_ID = "google/gemma-2b-it"

        llm = HuggingFaceEndpoint(
            repo_id=HUGGING_FACE_MODEL_ID,
            huggingfacehub_api_token=api_key,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
        )

        # 4. 프롬프트 템플릿 정의
        template = """당신은 '모구' 서비스에 대한 질문에 답변하는 친절한 AI 어시스턴트입니다.
        제공된 컨텍스트 정보만을 사용하여 사용자의 질문에 답변해 주세요.
        응답은 반드시 한국어로 해 주세요.

        컨텍스트:
        {context}

        질문:
        {question}

        답변:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 5. RAG 체인 구성
        rag_chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        # 🚫 st.error()를 호출하는 대신, 예외를 발생시켜 web.py에 알립니다.
        raise RagChainInitializationError(f"RAG 체인 생성 중 오류 발생: {e}")