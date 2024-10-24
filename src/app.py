# src/api.py
import os
import pinecone
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers, OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import ServerlessSpec,Pinecone
from src.helper import download_hugging_face_embeddings
from src.prompt import prompt_template

from pinecone import ServerlessSpec,Pinecone

load_dotenv()

# Get the Pinecone API key from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')

embeddings = download_hugging_face_embeddings()

# Initializing Pinecone
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

# Loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Initialize your template with dynamic content
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs = {"prompt": PROMPT}

llm=CTransformers(model="meta-llama/Llama-3.2-1B-Instruct",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

# llm = OpenAI()

# Initialize the chain with the LLM and the retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)