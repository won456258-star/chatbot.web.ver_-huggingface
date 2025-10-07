# 파일명: rag_logic.py (Hugging Face 환경용 최종 수정)

import streamlit as st
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ✅ 수정: itemgetter를 임포트하여 데이터 흐름을 명확하게 제어합니다.
from operator import itemgetter

# --- Hugging Face 환경 변수 확인 ---
def check_hf_api_token():
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
        # 2. 임베딩 모델 설정
        # 로컬에서 실행되며 안정적인 SentenceTransformerEmbeddings를 사용합니다.
        embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # 3. FAISS 벡터 저장소 로드
        vectorstore = FAISS.load_local(
            "my_faiss_db", 
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever()
        
        # 4. LLM 초기화
        HUGGING_FACE_MODEL_ID = "google/gemma-2b-it" 
        
        llm = HuggingFaceHub(
            repo_id=HUGGING_FACE_MODEL_ID,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
        )

        # 5. 프롬프트 템플릿 정의
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

        # 6. RAG 체인 구성 (✅ 최종 수정된 LCEL 구조)
        # itemgetter를 사용하여 retriever가 항상 문자열을 입력받도록 보장합니다.
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