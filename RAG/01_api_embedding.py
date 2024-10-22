import os

import openai
from dotenv import load_dotenv
from sentence_transformers import SimilarityFunction

# .env 파일 로드
load_dotenv()

# 환경 변수에서 OPENAI_API_KEY 값을 불러옴
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI API 키 설정
openai.api_key = api_key

# 유사도 계산 함수 설정(cosine))
similarity_fn = SimilarityFunction.to_similarity_fn("cosine")

# 텍스트 임베딩 함수 정의
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        text = text.strip() # 앞뒤 공백 제거
        response = openai.embeddings.create(input=[text], model=model)
        return response.data[0].embedding  # 첫 번째 임베딩 결과 반환
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None

# 예시 텍스트에 대한 임베딩 생성
embedding_1 = get_embedding("날씨가 화창하다.")
embedding_2 = get_embedding("날씨가 화창하다.")
embedding_3 = get_embedding("비가 오고 있다.")

if embedding_1 and embedding_2 and embedding_3:
    # 임베딩 간 유사도 계산
    score_1_2 = similarity_fn(embedding_1, embedding_2)
    score_1_3 = similarity_fn(embedding_1, embedding_3)
    score_2_3 = similarity_fn(embedding_2, embedding_3)

    # 유사도 출력
    print(f"embedding_1과 embedding_2의 유사도: {score_1_2}")
    print(f"embedding_1과 embedding_3의 유사도: {score_1_3}")
    print(f"embedding_2과 embedding_3의 유사도: {score_2_3}")
else:
    print("임베딩 생성에 실패하였습니다.")
