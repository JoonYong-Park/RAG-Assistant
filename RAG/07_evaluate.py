from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 문서 리스트 설정
doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다."
]

# 정답 데이터 (gold standard) 설정
gold_data = {
    "코로나가 유행한 연도": [0],
    "2022년 GDP 대비 R&D 예산": [2],
    "2024년의 국내총생산 전망": [1]
}

# BM25Retriever 설정
bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": 1}] * len(doc_list)
)
bm25_retriever.k = 1

# OpenAI Embeddings 설정
embedding = OpenAIEmbeddings(api_key=api_key)

# FAISS 벡터 스토어 생성 및 설정
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": i} for i in range(len(doc_list))]
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# EnsembleRetriever 설정
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.2, 0.8]
)

# 검색된 결과 가져오기
retrieved_docs = {query: ensemble_retriever.invoke(query) for query in gold_data}

# 검색 결과 평가 함수
def evalute_search(retrieved_docs, gold_standard, documents):
    precisions = []
    recalls = []
    f1s = []

    for query in gold_standard:
        retrieved = [doc.metadata['source'] for doc in retrieved_docs[query]]
        gold = gold_standard[query]

        y_true = [1 if i in gold else 0 for i in range(len(documents))]
        y_pred = [1 if i in retrieved else 0 for i in range(len(documents))]

        # 정확도, 재현율, F1 점수 계산
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # 평균 점수 계산
    avg_precision = sum(precisions) / len(gold_standard)
    avg_recall = sum(recalls) / len(gold_standard)
    avg_f1 = sum(f1s) / len(gold_standard)

    return avg_precision, avg_recall, avg_f1

# 평가 실행
avg_precision, avg_recall, avg_f1 = evalute_search(retrieved_docs, gold_data, doc_list)

# 결과 출력
print(f"precision: {avg_precision}")
print(f"recall: {avg_recall}")
print(f"f1: {avg_f1}")
