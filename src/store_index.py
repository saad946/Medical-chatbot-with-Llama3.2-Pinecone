import os
import logging
from dotenv import load_dotenv
import pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf, text_splitter, download_hugging_face_embeddings, create_index

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_pinecone_index(index_name):
    """Setup Pinecone index with specified name."""
    try:
        create_index(index_name)
    except Exception as e:
        logging.error(f"Failed to create index {index_name}: {e}")
        raise

def process_documents_and_store_vectors(directory, index_name):
    """Process documents from a directory and store their vector representations."""
    try:
        extracted_documents = load_pdf(directory)
        text_chunks = text_splitter(extracted_documents)
        embeddings = download_hugging_face_embeddings()
        
        if not text_chunks:
            logging.warning("No text chunks extracted from documents.")
            return
        
        docsearch = PineconeVectorStore.from_texts(
            [t.page_content for t in text_chunks],
            embeddings,
            index_name=index_name
        )
        logging.info("Successfully processed documents and stored vectors.")
    except Exception as e:
        logging.error(f"Error processing documents: {e}")
        raise

def main():
    load_dotenv()
    index_name = "medical-chatbot"

    # Setting up Pinecone index
    setup_pinecone_index(index_name)

    # Assuming the PDFs are stored in a directory called "data"
    process_documents_and_store_vectors("data", index_name)

if __name__ == "__main__":
    main()
