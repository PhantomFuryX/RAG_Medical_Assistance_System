import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from src.retrieval.document_retriever import MedicalDocumentRetriever

def load_documents(data_dir="../data/medical_texts"):
    """Load documents from the specified directory."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created directory {data_dir}. Please add medical files to this directory.")
        print("Example: Add .pdf or .txt files containing medical information.")
        return []
    
    documents = []
    try:
        # Load PDF files
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        print(f"Loaded {len(pdf_docs)} PDF documents from {data_dir}")
        documents.extend(pdf_docs)
        
        # Load text files
        txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        print(f"Loaded {len(txt_docs)} text documents from {data_dir}")
        documents.extend(txt_docs)
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return documents  # Return any documents that were loaded before the error

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for better embedding."""
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs

def main():
    # Load documents
    documents = load_documents()
    if not documents:
        print("No documents found. Please add PDF or text documents to the data directory.")
        return
    
    # Split documents into chunks
    split_docs = split_documents(documents)
    
    # Create the retriever and build the index
    retriever = MedicalDocumentRetriever(index_path="faiss_medical_index")
    retriever.create_index(split_docs)
    print("Index created successfully!")
    
    # Test the retriever
    query = "What are the symptoms of pneumonia?"
    results = retriever.retrieve(query)
    print("\nTest query results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)

if __name__ == "__main__":
    main()
