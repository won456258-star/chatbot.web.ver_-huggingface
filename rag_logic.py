# 파일명: rag_logic.py (최종 수정 버전)

import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# ✅ 변경점: @st.cache_resource 데코레이터가 인자를 받을 수 있도록 수정합니다.
# 이렇게 하면 API 키가 변경될 때만 함수가 다시 실행됩니다.
@st.cache_resource
def get_rag_chain(api_key: str):
    """
    Hugging Face API 키를 인자로 받아 RAG 체인을 생성하고 반환합니다.
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
            # ✅ 변경점: huggingfacehub_api_token 인자에 전달받은 키를 사용합니다.
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
        st.error(f"RAG 체인 초기화 중 치명적인 오류 발생: {e}")
        return None