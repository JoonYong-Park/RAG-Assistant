import os

import openai
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# .env 파일 로드
load_dotenv()

# 환경 변수에서 OPENAI_API_KEY 값을 불러옴
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI API 키 설정
openai.api_key = api_key

# OpenAI 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(api_key=api_key)

# PDF 로드
loader = PyPDFLoader("./data/car_to_car_sample.pdf")
documents = loader.load()  # 전체 페이지 가져오기

# 텍스트 청킹
text_splitter = CharacterTextSplitter(
    # separator=' .\n',
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

texts = text_splitter.split_documents(documents)
print(texts)

# 벡터 스토어에 텍스트 저장
db = FAISS.from_documents(texts, embedding_model)

# 리트리버 생성
retriever = db.as_retriever()
docs = retriever.invoke("뒤에도계속하여시속 40~50km로 교차로를진입하여들어오는 A 승용차를발견하고도 A가")

print(docs)
