# 파일명: create_db.py

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 1. 원본 데이터 파일 경로
#    - RAG의 학습 자료가 되는 텍스트 파일을 지정합니다.
#    - 파일이 없다면, data.txt 이름으로 새로 만들고 내용을 채워주세요.
DATA_PATH = "data.txt" 
DB_FAISS_PATH = "my_faiss_db"

def create_vector_db():
    """
    원본 텍스트 파일을 읽어와 FAISS 벡터 데이터베이스를 생성하고 저장합니다.
    """
    # 2. 데이터 로드
    try:
        loader = TextLoader(DATA_PATH, encoding="utf-8")
        documents = loader.load()
        print("✅ 원본 문서를 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"🚨 오류: '{DATA_PATH}' 파일을 찾을 수 없거나 읽을 수 없습니다. 파일이 정확한 위치에 있는지, 내용은 채워져 있는지 확인해주세요.")
        print(f"   원본 오류: {e}")
        return

    # 3. 텍스트 분할
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"✅ 문서를 {len(docs)}개의 조각으로 분할했습니다.")

    # 4. 임베딩 모델 설정
    model_name = "jhgan/ko-sroberta-multitask"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print("✅ 임베딩 모델을 준비했습니다.")

    # 5. FAISS 벡터 DB 생성 및 저장
    print("⏳ FAISS 벡터 데이터베이스를 생성하는 중입니다. 시간이 다소 걸릴 수 있습니다...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    print(f"🎉 성공! '{DB_FAISS_PATH}' 폴더에 새로운 벡터 데이터베이스를 저장했습니다.")

if __name__ == "__main__":
    # 기존 my_faiss_db 폴더가 있다면 삭제
    if os.path.exists(DB_FAISS_PATH):
        print(f"기존 '{DB_FAISS_PATH}' 폴더를 삭제합니다.")
        import shutil
        shutil.rmtree(DB_FAISS_PATH)
    
    create_vector_db()