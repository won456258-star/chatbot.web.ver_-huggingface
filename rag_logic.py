# 파일명: rag_logic.py (Hugging Face 환경용 최종 버전)

import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings # Hugging Face 임베딩
from langchain_community.llms import HuggingFaceHub # Hugging Face LLM 추론 API
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# --- Hugging Face 환경 변수 확인 ---
def check_hf_api_token():
    """HUGGINGFACEHUB_API_TOKEN 환경 변수가 로드되었는지 확인"""
    # os.environ은 .env 파일 (로컬) 또는 Streamlit Secrets (클라우드) 모두 처리
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        return True
    else:
        return False

@st.cache_resource
def get_rag_chain():
    # 1. API 토큰 로드 확인
    if not check_hf_api_token():
        return None

    try:
        # 2. 임베딩 모델 설정 (Hugging Face 임베딩 모델 사용)
        # 이 모델로 'my_faiss_db'를 미리 생성해야 합니다.
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 3. FAISS 벡터 저장소 재로드 
        vectorstore = FAISS.load_local(
            "my_faiss_db", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever()
        
        # 4. LLM 초기화 (Hugging Face Hub API 사용)
        # 모델 ID는 무료 추론이 지원되는 경량 모델을 사용합니다.
        HUGGING_FACE_MODEL_ID = "google/gemma-2b-it" 
        
        llm = HuggingFaceHub(
            repo_id=HUGGING_FACE_MODEL_ID,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
        )

        # 5. RAG 체인 구성 (LCEL 구조 유지)
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

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain

    except Exception as e:
        # 자세한 오류 로그를 출력하고 사용자에게 안내
        st.error(f"RAG 체인 초기화 중 치명적인 오류 발생: {e}")
        return None