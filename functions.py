from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader ##loading the document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
## to split the text into chunks 
##from langchain_community.embeddings.bedrock import BedrockEmbeddings

from langchain_huggingface import HuggingFaceEndpointEmbeddings


## for creating embeddings

from langchain_community.vectorstores.chroma import Chroma
# building chroma DB
DATA_PATH = "data"
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


##print(documents[0]) 
## Loading and reading pdf document

##Now Split the document into chunks for embedding process

def split_documents(documents : list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 80,
        chunk_overlap = 80,
        is_separator_regex = False
    )
    return text_splitter.split_documents(documents)

documents = load_documents()

## embedding function 

def get_embedding_functions():
    model = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEndpointEmbeddings(
    model=model,
    task="feature-extraction",
    huggingfacehub_api_token="HUGGINGFACE_API_KEY",
)  
    return embeddings




