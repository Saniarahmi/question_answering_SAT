from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
from typing import List

load_dotenv()

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere with embedding
    text = re.sub(r'[^\w\s\-.,;:()[\]{}\"\'/@#%&*+=<>?!]', ' ', text)
    # Normalize spacing around punctuation
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    return text.strip()

def process_pdfs_with_better_chunking():
    """Process PDFs with improved chunking strategy"""
    
    folder_path = "source_data"
    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} tidak ditemukan!")
        return
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("Tidak ada file PDF yang ditemukan!")
        return
    
    print(f"Ditemukan {len(pdf_files)} file PDF")
    
    # Initialize document storage
    all_documents = []
    
    # Process each PDF
    for pdf_file in pdf_files:
        try:
            file_path = os.path.join(folder_path, pdf_file)
            print(f"\nMemproses: {pdf_file}")
            
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Clean and enhance metadata
            for doc in documents:
                # Clean text content
                doc.page_content = clean_text(doc.page_content)
                
                # Add source information to metadata
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                
                doc.metadata.update({
                    'source_file': pdf_file,
                    'source_type': 'SAMSAT_Document',
                    'processed': True
                })
                
                # Only add non-empty documents
                if len(doc.page_content.strip()) > 50:  # Minimum content length
                    all_documents.append(doc)
            
            print(f"  - Berhasil memuat {len(documents)} halaman")
            print(f"  - Valid documents: {len([d for d in documents if len(d.page_content.strip()) > 50])}")
            
        except Exception as e:
            print(f"  - Error: {e}")
            continue
    
    if not all_documents:
        print("Tidak ada dokumen valid yang dapat diproses!")
        return
    
    print(f"\nTotal dokumen valid: {len(all_documents)}")
    
    # Advanced text splitting strategy
    text_splitters = [
        # Primary splitter - larger chunks with overlap
        RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased chunk size
            chunk_overlap=300,  # Significant overlap
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple newlines
                "\n\n",    # Double newlines  
                "\n",      # Single newlines
                ". ",      # Sentence endings
                "? ",      # Question endings
                "! ",      # Exclamation endings
                "; ",      # Semicolons
                ", ",      # Commas
                " ",       # Spaces
                ""         # Characters
            ]
        ),
        # Secondary splitter for very long sections
        RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len
        )
    ]
    
    # Apply text splitting
    processed_chunks = []
    
    for splitter in text_splitters:
        if not processed_chunks:  # Use first splitter
            processed_chunks = splitter.split_documents(all_documents)
        else:  # Apply additional splitting if chunks are still too large
            large_chunks = [chunk for chunk in processed_chunks if len(chunk.page_content) > 1500]
            if large_chunks:
                additional_splits = splitter.split_documents(large_chunks)
                # Replace large chunks with their splits
                processed_chunks = [chunk for chunk in processed_chunks if len(chunk.page_content) <= 1500]
                processed_chunks.extend(additional_splits)
    
    # Filter out very short chunks
    processed_chunks = [chunk for chunk in processed_chunks if len(chunk.page_content.strip()) > 100]
    
    print(f"Jumlah chunks setelah pemrosesan: {len(processed_chunks)}")
    
    # Show sample chunks for verification
    print("\nSample chunks:")
    for i, chunk in enumerate(processed_chunks[:3]):
        print(f"\nChunk {i+1} (length: {len(chunk.page_content)}):")
        print(f"Content preview: {chunk.page_content[:200]}...")
        print(f"Metadata: {chunk.metadata}")
    
    # Initialize better embeddings
    try:
        print("\nMenginisialisasi embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("Embeddings berhasil diinisialisasi")
        
        # Test embedding
        test_text = "SAMSAT adalah layanan administrasi kendaraan bermotor"
        test_embedding = embeddings.embed_query(test_text)
        print(f"Test embedding dimension: {len(test_embedding)}")
        
    except Exception as e:
        print(f"Error inisialisasi embeddings: {e}")
        return
    
    # Create and persist vector store
    try:
        print("\nMembuat vector store...")
        
        # Remove existing data directory if it exists
        import shutil
        if os.path.exists("data"):
            shutil.rmtree("data")
            print("Menghapus vector store lama")
        
        # Create new vector store
        vectorstore = Chroma.from_documents(
            documents=processed_chunks,
            embedding=embeddings,
            persist_directory="data",
            collection_metadata={"description": "SAMSAT Sorong Knowledge Base"}
        )
        
        print(f"Vector store berhasil dibuat dengan {len(processed_chunks)} chunks")
        
        # Test the vector store
        print("\nTesting vector store...")
        test_queries = [
            "SAMSAT",
            "SIM",
            "pajak kendaraan",
            "denda keterlambatan",
            "syarat perpanjangan"
        ]
        
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": 0.3}
        )
        
        for query in test_queries:
            results = retriever.invoke(query)
            print(f"\nQuery: '{query}' -> {len(results)} results found")
            if results:
                print(f"  Top result preview: {results[0].page_content[:100]}...")
        
        print("\n✅ Vector store berhasil dibuat dan ditest!")
        print("Anda sekarang dapat menjalankan aplikasi Flask")
        
    except Exception as e:
        print(f"Error membuat vector store: {e}")
        return

def analyze_existing_vectorstore():
    """Analyze existing vectorstore to debug issues"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        vectorstore = Chroma(
            persist_directory="data",
            embedding_function=embeddings
        )
        
        # Get collection info
        collection = vectorstore._collection
        print(f"Collection count: {collection.count()}")
        
        # Test various queries
        test_queries = [
            "SAMSAT",
            "SIM", 
            "pajak kendaraan",
            "Sorong",
            "administrasi",
            "biaya",
            "syarat",
            "dokumen"
        ]
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        for query in test_queries:
            results = retriever.invoke(query)
            print(f"\nQuery: '{query}'")
            print(f"Results found: {len(results)}")
            
            if results:
                for i, result in enumerate(results[:2]):
                    print(f"  Result {i+1}: {result.page_content[:150]}...")
            else:
                print("  No results found!")
                
    except Exception as e:
        print(f"Error analyzing vectorstore: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        print("Analyzing existing vectorstore...")
        analyze_existing_vectorstore()
    else:
        print("Processing PDFs and creating vectorstore...")
        process_pdfs_with_better_chunking()