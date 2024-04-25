import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Step 1: Load the document
loader = PyPDFLoader("2404.07143.pdf")
data = loader.load()

# Step 2: Split the document into chunks
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Step 3: Create Chunks Embedding
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="API KEY", model="models/embedding-001")

# Step 4: Store the chunks in vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()

# Step 5: Setup the Vector Store as a Retriever
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Step 6: Define RAG chain
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a Helpful AI Bot. 
    You take the context and question from the user. Your answer should be based on the specific context."""),
    HumanMessagePromptTemplate.from_template("""Answer the question based on the given context.
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer: """)
])

chat_model = ChatGoogleGenerativeAI(google_api_key="API KEY", model="gemini-1.5-pro-latest")
output_parser = StrOutputParser()

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
st.title("RAG System on 'Leave No Context Behind' Paper")

# User input
user_input = st.text_input("Enter your question:", "")

# Generate response
if st.button("Generate Response"):
    if user_input:
        response = rag_chain.invoke(user_input)
        st.markdown(response)
