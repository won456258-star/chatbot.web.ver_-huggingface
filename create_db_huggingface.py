# 파일명: create_db_huggingface.py

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import time

def create_and_store_db_hf():
    # 'my_data.txt' 파일을 로드합니다.
    print("✅ 'my_data.txt' 파일을 로드합니다.")
    loader = TextLoader("my_data.txt", encoding="utf-8")
    documents = loader.load()

    # 문서를 청크 단위로 분할합니다.
    print("\n✅ 문서를 청크 단위로 분할합니다.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"총 {len(docs)}개의 문서 조각으로 분할되었습니다.")

    # Hugging Face 임베딩 모델을 초기화합니다.
    print("\n✅ Hugging Face 임베딩 모델을 초기화합니다. (sentence-transformers/all-MiniLM-L6-v2)")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = SentenceTransformerEmbeddings(model_name=model_name)
    
    # FAISS 벡터 DB를 생성하고 저장합니다.
    print("\n✅ FAISS 벡터 데이터베이스를 생성하고 저장합니다. (시간이 걸릴 수 있습니다)")
    start_time = time.time()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("my_faiss_db")
    end_time = time.time()
    
    print(f"\n🎉🎉🎉 성공! 'my_faiss_db' 폴더가 성공적으로 생성되었습니다. (소요 시간: {end_time - start_time:.2f}초)")

if __name__ == '__main__':
    create_and_store_db_hf()