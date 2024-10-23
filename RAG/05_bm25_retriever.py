import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

# .env 파일에서 API 키 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

# BM25Retriever 사용
bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": 1}] * len(doc_list)
)
bm25_retriever.k = 1

# OpenAI Embeddings 객체 생성
embedding = OpenAIEmbeddings(api_key=api_key)

# FAISS 벡터 스토어 생성 및 설정
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": 1}] * len(doc_list)
)

# FAISS 검색기 생성
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# 질의 실행
query = "2022년 우리나라 GDP대비 R&D 규모는?"

# BM25와 FAISS 검색 결과 실행
bm25_docs = bm25_retriever.invoke(query)
faiss_docs = faiss_retriever.invoke(query)

# 결과 출력
print('-----------')
print("BM25 검색 결과:", bm25_docs)
print('-----------')
print("FAISS 검색 결과:", faiss_docs)
