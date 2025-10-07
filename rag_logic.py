# 파일명: rag_logic.py

import streamlit as st
# ❗ [수정] 최신 임베딩 클래스를 import 합니다.
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ❗ [수정] RunnablePassthrough를 import 합니다.
from langchain_core.runnables import RunnablePassthrough

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
        # 1. 임베딩 모델 설정 (최신 클래스로 변경)
        # ❗ [수정] SentenceTransformerEmbeddings -> HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
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

        # 5. RAG 체인 구성 (RunnablePassthrough 사용)
        # ❗ [수정] itemgetter 대신 RunnablePassthrough를 사용하여 더 명확하게 구성합니다.
        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        # 🚫 st.error()를 호출하는 대신, 예외를 발생시켜 web.py에 알립니다.
        raise RagChainInitializationError(f"RAG 체인 생성 중 오류 발생: {e}")