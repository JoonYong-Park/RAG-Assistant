import os

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)
model = "gpt-4"

# 문서 리스트
doc_list = [
    "우리나라는 2022년에 코로나가 유행했다.",
    "우리나라 2024년 GDP 전망은 3.0%이다.",
    "우리나라는 2022년 국내총생산 중 연구개발 예산은 약 5%이다.",
    "삼성전자 2024년 1분기 매출액은 약 7조원으로 잠정 추정됩니다.",
    "2024년 7월 19일 삼성전자 주가는 64,500원입니다."
]

# BM25 리트리버 설정
bm25_retriever = BM25Retriever.from_texts(
    doc_list, metadatas=[{"source": i} for i in range(len(doc_list))]
)
bm25_retriever.k = 1

# FAISS 벡터 스토어 생성
embedding = OpenAIEmbeddings(api_key=api_key)
faiss_vectorstore = FAISS.from_texts(
    doc_list, embedding, metadatas=[{"source": i} for i in range(len(doc_list))]
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

# 앙상블 리트리버 설정
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.2, 0.8]
)

# 검색 함수
def search(query):
    ensemble_docs = ensemble_retriever.invoke(query)
    return ensemble_docs

# ChatGPT 생성 함수
def chatgpt_generate(query):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": query
        }
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

# 검색된 문서를 기반으로 ChatGPT 응답 생성
def prompt_and_generate(query, docs):
    prompt = f"""아래 질문을 기반으로 검색된 문서를 참고하여 질문에 대한 답변을 생성하시오.\n\n질문: {query}\n"""
    for i, doc in enumerate(docs):
        prompt += f"문서{i+1}: {doc}\n"

    print("-"*100)
    print(prompt)

    answer = chatgpt_generate(prompt)
    return answer

# 검색 및 결과 생성
query = "삼성전자의 2024sus 1분기 매출액은?"
# query = "삼성전자의 올해 매출액은?"
retrieved = [doc.page_content for doc in search(query)]
answer = prompt_and_generate(query, retrieved)

print("-"*100)  
print("답변: ", answer)
