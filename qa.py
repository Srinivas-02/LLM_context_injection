from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.prompts import ChatPromptTemplate
import torch
import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configuration
DATA_PATH = "data"
MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
CUSTOM_RAG_PROMPT = """You are a helpful chat bot for a woodworking furniture website. 
Your purpose is to assist customers with questions about wooden furniture, carpentry techniques, 
and product information. Use the following context to answer the question. 
If you don't know the answer, politely say you don't know and suggest contacting the support team.

Context: {context}
Question: {question}

Helpful Answer:"""
def load_and_process_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

def create_or_load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    if not os.path.exists(FAISS_INDEX_PATH):
        # Create new index if it doesn't exist
        print("Creating new FAISS index...")
        texts = load_and_process_documents()
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
    else:
        # Load existing index
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_qa_chain():
    # Initialize model pipeline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    hf_pipeline = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        torch_dtype=torch.float32
    )
    
    # Load/create vector store
    vector_store = create_or_load_vector_store()
    
    # Create RAG chain
    prompt = ChatPromptTemplate.from_template(CUSTOM_RAG_PROMPT)

    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | HuggingFacePipeline(pipeline=hf_pipeline)
        | StrOutputParser()
    )

    
    return qa_chain

def main():
    qa_chain = initialize_qa_chain()
    
    print("Furniture Bot: Hello! Ask me about our products or services.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        
        if not query:
            continue
            
        try:
            response = qa_chain.invoke(query)
            print(f"\nFurniture Bot: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()