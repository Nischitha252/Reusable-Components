import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import openai  # Ensure you have your openai configuration set
import PyPDF2
import docx

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def read_docx(file):
    doc = docx.Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

def read_file(file):
    file_name = file.name.lower()
    if file_name.endswith('.pdf'):
        return read_pdf(file)
    elif file_name.endswith('.docx'):
        return read_docx(file)
    else:
        return None

def process_documents(doc1_text, doc2_text, embeddings):
    # Prepare documents with source labels.
    docs = [
        {"source": "Document 1", "text": doc1_text},
        {"source": "Document 2", "text": doc2_text}
    ]
    
    # Split each document into chunks using a recursive character text splitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            splitted_docs.append({"source": doc["source"], "text": chunk})
    
    # Create a FAISS index using the provided Azure OpenAI embeddings.
    texts = [d["text"] for d in splitted_docs]
    metadatas = [{"source": d["source"]} for d in splitted_docs]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

def main():
    st.title("Document Comparison using AzureOpenAIEmbeddings, FAISS & AzureChatOpenAI")
    st.write("Upload your two document files (PDF or DOCX):")
    
    file1 = st.file_uploader("Upload Document 1", type=["pdf", "docx"])
    file2 = st.file_uploader("Upload Document 2", type=["pdf", "docx"])
    
    if st.button("Process Documents"):
        if file1 is None or file2 is None:
            st.error("Please upload both document files.")
        else:
            # Read the content of the uploaded files.
            doc1_text = read_file(file1)
            doc2_text = read_file(file2)
            
            if doc1_text is None or doc2_text is None:
                st.error("Unsupported file format. Please upload a PDF or DOCX file.")
                return

            # Initialize your custom Azure OpenAI embeddings.
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint="",
                api_key="",
                azure_deployment="",  # Replace with your actual deployment name
                openai_api_version=""
            )
            
            vectorstore = process_documents(doc1_text, doc2_text, embeddings)
            st.session_state.vectorstore = vectorstore
            st.success("Documents processed and FAISS index built. Now you can enter your query.")
    
    # Allow the user to enter a custom query once the FAISS index is ready.
    if "vectorstore" in st.session_state:
        user_query = st.text_input("Enter your query:")
        if st.button("Generate Answer"):
            if not user_query:
                st.error("Please enter a query.")
            else:
                # Initialize your custom Azure Chat OpenAI LLM.
                llm = AzureChatOpenAI(
                    azure_deployment="",  # Replace with your actual deployment name
                    openai_api_key="",
                    openai_api_version="",
                    azure_endpoint=""
                )
                # Build a RetrievalQA chain using the FAISS index as the retriever.
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                )
                result = qa_chain.invoke(user_query)
                
                # Optionally, display the retrieved passages.
                retrieved_docs = st.session_state.vectorstore.similarity_search(user_query, k=5)
                st.subheader("Retrieved Passages")
                for doc in retrieved_docs:
                    st.markdown(f"**{doc.metadata.get('source', 'Document')}**: {doc.page_content}")
                    
                st.subheader("Answer")
                st.write(result)

if __name__ == "__main__":
    main()