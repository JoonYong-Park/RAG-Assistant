from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# PDF 파일 로드 및 텍스트 추출
loader = PyPDFLoader("./data/car_to_car_sample.pdf")
pages = loader.load_and_split()

# 첫 번째 페이지의 텍스트 추출
text = pages[0].page_content

print(f"\n\n- 텍스트 길이: {len(text)}")
print(f"\n\n- 텍스트:\n{text}")

# 텍스트 청킹 설정
text_splitter = CharacterTextSplitter(
    separator=' .\n',   # 구분자
    chunk_size=500,     # 청킹 사이즈
    chunk_overlap=100,  # 청킹 간 겹침 부분
    length_function=len # 길이 함수
)

# 텍스트 청킹
texts = text_splitter.split_text(text)

# 청킹 결과 출력
print("-"*50)
print(f"\n\n- 청킹된 갯수: {len(texts)}")

# 첫 두 개의 청킹 결과 출력 (반복문 사용)
for i in range(2):  # 처음 두 개의 청킹된 텍스트만 출력
    print(f"\n\n- 청킹한 텍스트 {i+1}번의 길이: {len(texts[i])}")
    print(f"\n\n- 청킹한 텍스트 {i+1}:\n {texts[i]}")
    print("-"*50)
