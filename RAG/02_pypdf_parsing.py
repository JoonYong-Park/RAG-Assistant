from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./data/car_to_car_sample.pdf")
pages = loader.load_and_split()

text = pages[0].page_content
print(text)