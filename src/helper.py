from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from pinecone import ServerlessSpec,Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


# Load the .env file
load_dotenv()

# Get the Pinecone API key from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')

#Extract data from the PDF
def load_pdf(directory):
    # Initialize the PDF loader with the directory containing PDF files
    loader = PyPDFDirectoryLoader(directory)
    
    # Load the data
    documents = loader.load()

    return documents


#Create text chunks
def text_splitter(extracted_documents):
    text_splitter= RecursiveCharacterTextSplitter(
    chunk_size=600,         
    chunk_overlap=80,      
    length_function=len,
    
        
)
    text_chunks= text_splitter.split_documents(extracted_documents) 
    
    return text_chunks

    
#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

pc = Pinecone(api_key=pinecone_api_key)

def create_index(index_name):
    pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


