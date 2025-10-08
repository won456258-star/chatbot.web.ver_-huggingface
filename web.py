import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 허깅페이스 API 토큰 설정
# Streamlit Secrets나 직접 코드를 통해 설정할 수 있습니다.
# 이 예제에서는 .env 파일을 사용합니다.
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 페이지 설정
st.set_page_config(page_title="PDF 문서 기반 챗봇", layout="wide")
st.title("📄 PDF 문서와 대화하는 챗봇")

# 사이드바 설정
with st.sidebar:
    st.header("PDF 파일 업로드")
    uploaded_file = st.file_uploader("여기에 PDF 파일을 업로드하세요.", type="pdf")
    st.info("PDF를 업로드하면 내용 처리가 시작됩니다.")

# 벡터 저장소를 세션 상태에 초기화
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 대화 기록을 세션 상태에 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 함수: PDF에서 텍스트를 추출하고 청크로 분할
def process_pdf(file):
    if file is not None:
        # 임시 파일로 저장
        temp_file_path = f"./temp_{file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())

        # PDF 로더
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # 임시 파일 삭제
        os.remove(temp_file_path)
        return chunks
    return None

# 함수: 임베딩 및 벡터 저장소 생성
def create_vector_store(chunks):
    if chunks:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embedding=embeddings)
        return vector_store
    return None

# PDF 파일이 업로드되면 처리 시작
if uploaded_file is not None and st.session_state.vector_store is None:
    with st.spinner("PDF 파일을 처리 중입니다... 잠시만 기다려주세요."):
        # 1. PDF 처리
        text_chunks = process_pdf(uploaded_file)
        
        # 2. 벡터 저장소 생성
        if text_chunks:
            st.session_state.vector_store = create_vector_store(text_chunks)
            st.success("PDF 파일 처리가 완료되었습니다! 이제 질문을 시작할 수 있습니다.")
        else:
            st.error("PDF 파일에서 텍스트를 추출하지 못했습니다.")

# 이전 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
user_question = st.chat_input("PDF 내용에 대해 질문해보세요.")

if user_question and st.session_state.vector_store:
    # 사용자 질문을 대화 기록에 추가
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # LLM 모델 초기화
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.1, "max_length": 1024}
    )

    # 프롬프트 템플릿 정의 (최신 방식)
    template = """
    당신은 친절한 AI 어시스턴트입니다. 주어진 문맥(context) 정보를 바탕으로 사용자의 질문(input)에 대해 답변해주세요.
    
    [Context]:
    {context}
    
    [Question]:
    {input}
    
    [Answer]:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # RAG 체인 생성 (최신 방식)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 챗봇 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중입니다..."):
            try:
                # 체인 호출
                result = retrieval_chain.invoke({"input": user_question})
                response = result.get("answer", "죄송합니다, 답변을 생성하지 못했습니다.")
                st.write(response)
                # 어시스턴트 응답을 대화 기록에 추가
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

elif user_question:
    st.warning("먼저 PDF 파일을 업로드해주세요.")