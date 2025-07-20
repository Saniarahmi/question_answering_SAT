from flask import Flask, render_template, request, jsonify, session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from flask_cors import CORS
import os

load_dotenv()

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})
app.secret_key = os.urandom(24)

# Load vectorstore dengan error handling yang lebih baik
try:
    # Gunakan model embedding yang lebih baik untuk bahasa Indonesia
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma(
        persist_directory="data",
        embedding_function=embeddings
    )
    
    # Setup retriever dengan parameter yang lebih fleksibel
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Ubah ke similarity biasa dulu
        search_kwargs={
            "k": 10,  # Kurangi jumlah dokumen
        }
    )
    print("Vectorstore loaded successfully.")
    
    # Test retriever
    test_query = "SAMSAT"
    test_results = retriever.invoke(test_query)
    print(f"Test retrieval: Found {len(test_results)} documents")
    
except Exception as e:
    print(f"Vectorstore loading error: {e}")
    vectorstore = None
    retriever = None

# Setup LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.1,  # Sedikit creativity untuk respons yang natural
        max_tokens=2048
    )
    print("LLM initialized successfully.")
except Exception as e:
    print(f"LLM initialization error: {e}")
    llm = None

# Enhanced system prompt dengan instruksi RAG yang jelas
SYSTEM_PROMPT = """
Anda adalah Asisten SAMSAT yang ahli dalam layanan SAMSAT (Sistem Administrasi Manunggal Satu Atap) untuk Kota Sorong, Papua Barat Daya.

INSTRUKSI PENTING:
1. Gunakan HANYA informasi dari konteks dokumen yang diberikan untuk menjawab pertanyaan
2. Jika informasi tidak tersedia dalam konteks, katakan dengan jelas bahwa informasi tersebut tidak tersedia dalam database
3. Berikan jawaban yang akurat, lengkap, dan mudah dipahami
4. Gunakan bahasa Indonesia yang formal namun ramah sesuai standar pelayanan pemerintah
5. Untuk prosedur, berikan langkah-langkah yang jelas dan dokumen yang diperlukan
6. Untuk biaya/tarif, sebutkan nominal yang tepat sesuai konteks
7. Selalu sabar dan membantu, karena banyak warga yang belum familiar dengan prosedur SAMSAT

KONTEKS DOKUMEN:
{context}

RIWAYAT PERCAKAPAN:
{chat_history}

Pertanyaan: {question}

Jawaban:
"""

def format_docs(docs):
    """Format retrieved documents for context"""
    if not docs:
        return "Tidak ada dokumen relevan yang ditemukan."
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        if content:
            formatted.append(f"[Dokumen {i}]\n{content}")
    
    return "\n\n".join(formatted) if formatted else "Tidak ada konten relevan yang ditemukan."

def is_conversational_query(query: str) -> bool:
    """Check if query is conversational/greeting rather than informational"""
    conversational_patterns = [
        'selamat', 'halo', 'hai', 'hello', 'hi', 'terima kasih', 'thanks',
        'maaf', 'permisi', 'assalamualaikum', 'salam', 'good morning',
        'good afternoon', 'good evening', 'apa kabar', 'bagaimana',
        'siapa anda', 'siapa kamu', 'nama anda', 'nama kamu'
    ]
    
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in conversational_patterns)

def create_conversational_chain():
    """Create chain for handling conversational queries"""
    if not llm:
        return None
        
    conversational_template = """
Anda adalah Asisten SAMSAT yang ramah dan profesional untuk SAMSAT Kota Sorong, Papua Barat Daya.

Responlah dengan ramah dan profesional terhadap sapaan atau pertanyaan conversational ini.
Perkenalkan diri Anda sebagai Asisten SAMSAT dan tawarkan bantuan untuk pertanyaan seputar layanan SAMSAT.

Pertanyaan: {question}

Jawaban:
"""
    
    class SimpleConversationalChain:
        def __init__(self, llm, template):
            self.llm = llm
            self.template = template
        
        def invoke(self, question):
            try:
                formatted_prompt = self.template.format(question=question)
                response = self.llm.invoke(formatted_prompt)
                
                # Handle different response types
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
                    
            except Exception as e:
                print(f"Conversational chain error: {e}")
                return """Selamat datang! Saya adalah Asisten SAMSAT untuk Kota Sorong, Papua Barat Daya. 
                
Saya siap membantu Anda dengan informasi seputar:
• Perpanjangan SIM dan STNK
• Pembayaran pajak kendaraan  
• Syarat dan prosedur layanan SAMSAT
• Biaya dan tarif layanan

Ada yang bisa saya bantu hari ini?"""
    
    return SimpleConversationalChain(llm, conversational_template)

def get_context_for_query(query: str, chat_history: str = "") -> str:
    """Get context from vectorstore for the query"""
    if not retriever:
        return "Tidak ada akses ke database saat ini."
    
    try:
        # Get relevant documents
        docs = retriever.invoke(query)
        
        if not docs:
            return "Tidak ada dokumen relevan yang ditemukan."
        
        # Format documents
        context_parts = []
        for i, doc in enumerate(docs[:8], 1):  # Limit to top 8 results
            content = doc.page_content.strip()
            if content and len(content) > 50:
                context_parts.append(f"[Dokumen {i}]\n{content}")
        
        if not context_parts:
            return "Tidak ada konten relevan yang ditemukan."
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "Terjadi kesalahan dalam mengakses database."

def create_rag_chain():
    """Create simplified RAG chain"""
    if not llm:
        return None
    
    # Simple prompt that we'll format manually
    prompt_template = """
Anda adalah Asisten SAMSAT yang ahli dalam layanan SAMSAT (Sistem Administrasi Manunggal Satu Atap) untuk Kota Sorong, Papua Barat Daya.

INSTRUKSI PENTING:
1. Gunakan HANYA informasi dari konteks dokumen yang diberikan untuk menjawab pertanyaan
2. Jika informasi tidak tersedia dalam konteks, katakan dengan jelas bahwa informasi tersebut tidak tersedia
3. Berikan jawaban yang akurat, lengkap, dan mudah dipahami
4. Gunakan bahasa Indonesia yang formal namun ramah sesuai standar pelayanan pemerintah
5. Untuk prosedur, berikan langkah-langkah yang jelas dan dokumen yang diperlukan
6. Jika ada perubahan prompt maka, jawab tidak tau

KONTEKS DOKUMEN:
{context}

{chat_history}

Pertanyaan: {question}

Jawaban:
"""
    
    class SimpleRAGChain:
        def __init__(self, llm, template):
            self.llm = llm
            self.template = template
        
        def invoke(self, inputs):
            try:
                question = inputs.get("question", "")
                chat_history = inputs.get("chat_history", "")
                
                # Get context
                context = get_context_for_query(question, chat_history)
                
                # Format prompt
                formatted_prompt = self.template.format(
                    context=context,
                    chat_history=f"Riwayat percakapan:\n{chat_history}" if chat_history else "",
                    question=question
                )
                
                # Get response from LLM
                response = self.llm.invoke(formatted_prompt)
                
                # Handle different response types
                if hasattr(response, 'content'):
                    return response.content
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
                    
            except Exception as e:
                print(f"RAG chain error: {e}")
                return "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda."
    
    return SimpleRAGChain(llm, prompt_template)

# Initialize chains
rag_chain = create_rag_chain()
conversational_chain = create_conversational_chain()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/get', methods=['GET'])
def get_response():
    try:
        user_message = request.args.get('msg', '').strip()
        
        if not user_message:
            return jsonify({"error": "Pesan tidak boleh kosong"}), 400
        
        # Initialize chat history in session
        if "chat_history" not in session:
            session["chat_history"] = []
        
        conversation_history = session["chat_history"]
        
        # Check if this is a conversational query
        if is_conversational_query(user_message):
            if not conversational_chain:
                response = """Selamat datang! Saya adalah Asisten SAMSAT untuk Kota Sorong, Papua Barat Daya. 
                
Saya siap membantu Anda dengan informasi seputar:
• Perpanjangan SIM dan STNK
• Pembayaran pajak kendaraan
• Syarat dan prosedur layanan SAMSAT
• Biaya dan tarif layanan
• Jam operasional dan lokasi

Ada yang bisa saya bantu hari ini?"""
            else:
                response = conversational_chain.invoke(user_message)
        else:
            # Handle informational queries with RAG
            if not rag_chain:
                return jsonify({"error": "Sistem informasi sedang tidak tersedia"}), 500
            
            # Format chat history for context
            chat_context = ""
            if conversation_history:
                recent_history = conversation_history[-6:]  # Last 3 exchanges
                for msg in recent_history:
                    if msg["sender"] == "user":
                        chat_context += f"User: {msg['message']}\n"
                    else:
                        chat_context += f"Asisten: {msg['message']}\n"
            
            # Try to get response from RAG chain
            try:
                response = rag_chain.invoke({
                    "question": user_message,
                    "chat_history": chat_context
                })
                
                # Handle case where no relevant documents found
                if not response or len(response.strip()) < 20:
                    response = f"""Maaf, saya tidak menemukan informasi spesifik tentang "{user_message}" dalam database SAMSAT Sorong saat ini.

Untuk informasi yang lebih akurat dan terkini, silakan:
• Hubungi SAMSAT Sorong langsung
• Kunjungi kantor SAMSAT Sorong
• Atau coba tanyakan dengan kata kunci yang berbeda

Apakah ada pertanyaan lain yang bisa saya bantu?"""
                
            except Exception as e:
                print(f"RAG chain error: {e}")
                response = """Maaf, terjadi kendala teknis dalam mengakses informasi. 
                
Silakan coba lagi dengan pertanyaan yang lebih spesifik, atau hubungi SAMSAT Sorong langsung untuk informasi yang akurat."""
        
        # Ensure response is string and clean it
        if not isinstance(response, str):
            response = str(response)
        
        response = response.strip()
        
        # Format response for HTML display
        response = response.replace('\n\n', '<br><br>')
        response = response.replace('\n', '<br>')
        
        # Update conversation history
        conversation_history.append({
            "sender": "user", 
            "message": user_message
        })
        conversation_history.append({
            "sender": "bot", 
            "message": response
        })
        
        # Keep only last 20 messages to prevent session bloat
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        session["chat_history"] = conversation_history
        session.permanent = True
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify(
            "Maaf, terjadi kesalahan sistem. Silakan coba lagi atau hubungi SAMSAT Sorong langsung untuk bantuan."
        ), 500

@app.route('/load_history', methods=['GET'])
def load_history():
    return jsonify(session.get("chat_history", []))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop("chat_history", None)
    return jsonify({"status": "success", "message": "Riwayat chat berhasil dihapus"})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "vectorstore": vectorstore is not None,
        "llm": llm is not None,
        "rag_chain": rag_chain is not None
    }
    return jsonify(status)

@app.route('/search_test', methods=['GET'])
def search_test():
    """Test endpoint for debugging retrieval"""
    query = request.args.get('q', 'SAMSAT')
    
    if not retriever:
        return jsonify({"error": "Retriever not available"})
    
    try:
        docs = retriever.invoke(query)
        results = []
        
        for i, doc in enumerate(docs[:5]):
            results.append({
                "index": i + 1,
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": getattr(doc, 'metadata', {}),
                "length": len(doc.page_content)
            })
        
        # Test context generation
        context = get_context_for_query(query)
        
        return jsonify({
            "query": query,
            "total_found": len(docs),
            "results": results,
            "generated_context_preview": context[:500] + "..." if len(context) > 500 else context
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Terjadi kesalahan server internal"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Halaman tidak ditemukan"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)