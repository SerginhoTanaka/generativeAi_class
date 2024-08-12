import os
import uuid
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

data_dir = "data"
loader = PyPDFDirectoryLoader(data_dir)
docs = loader.load()

# Create unique IDs for each document
for doc in docs:
    doc.metadata["id"] = str(uuid.uuid4())

# Initialize the embedding function
embeddings = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")

# Create a FAISS index
index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

# Initialize the FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Add documents to the FAISS vector store
vector_store.add_documents(docs)

# Check the number of documents in the FAISS collection
print("There are", len(vector_store.docstore._dict), "docs in the collection")

system_prompt = (
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer 
    the question. If you don't know the answer, say that you 
    don't know. Use three sentences maximum and keep the 
    answer concise.
    \n\n
    {context}
    """
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{text}")
])

faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rag_chain = RetrievalQA.from_llm(
    llm=ChatOpenAI(openai_api_key=api_key),
    retriever=faiss_retriever,
    return_source_documents=True
)

query = "O que o PDF attention is all you need fala?"
result = rag_chain({"query": query})  # Corrected method call

# Print the result
print(result['result'])  # Access the answer from the result