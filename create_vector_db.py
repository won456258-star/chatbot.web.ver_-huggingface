# 파일명: create_vector_db.py (API 키 사용 버전)
import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# .env 파일에서 환경 변수 로드
print("✅ .env 파일에서 환경 변수를 로드합니다.")
load_dotenv()

def create_and_store_db():
    # 1. 텍스트 파일 로드
    print("\n✅ 'my_data.txt' 파일을 로드합니다.")
    loader = TextLoader("my_data.txt", encoding="utf-8")
    documents = loader.load()

    # 2. 텍스트 분할
    print("\n✅ 문서를 청크 단위로 분할합니다.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 3. OpenAI 임베딩 모델 초기화
    print("\n✅ OpenAI 임베딩 모델을 초기화합니다.")
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        print(f"\n❌ 오류: OpenAIEmbeddings 초기화 실패. API 키 설정을 확인하세요. (상세: {e})")
        sys.exit(1)

    # 4. FAISS 벡터 DB 생성 및 저장
    print("\n✅ FAISS 벡터 데이터베이스를 생성하고 저장합니다.")
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("my_faiss_db")
        print("\n🎉🎉🎉 성공! 'my_faiss_db' 폴더가 성공적으로 생성되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류: FAISS DB 생성 실패. API 키가 유효한지 확인하세요. (상세: {e})")
        sys.exit(1)

if __name__ == '__main__':
    create_and_store_db()