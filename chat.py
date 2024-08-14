import os
from dotenv import load_dotenv
import faiss
from uuid import uuid4
from pprint import pprint
import streamlit as st

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader

class DocumentProcessor:
    def __init__(self, data_dir: str, api_key: str):
        self.data_dir = data_dir
        self.api_key = api_key
        self.loader = PyPDFDirectoryLoader(data_dir)
        self.documents = self.loader.load_and_split()

class VectorStoreManager:
    def __init__(self, documents, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("example query")))
        self.index_to_docstore_id = {i: str(uuid4()) for i in range(len(documents))}
        self.docstore = InMemoryDocstore({self.index_to_docstore_id[i]: doc for i, doc in enumerate(documents)})
        self.vector_store = FAISS(embedding_function=self.embeddings, index=self.index, 
                                 docstore=self.docstore, index_to_docstore_id=self.index_to_docstore_id)
        self.vector_store.add_documents(documents)

class QAChain:
    def __init__(self, vector_store: FAISS, api_key: str):
        self.llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
        self.vector_store = vector_store
        self.history_aware_retriever = self.create_history_aware_retriever()
        self.qa_chain = self.create_stuff_documents_chain()
        self.convo_qa_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)

    def create_history_aware_retriever(self):
        condense_question_system_template = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        condense_question_prompt = ChatPromptTemplate.from_messages([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        return create_history_aware_retriever(self.llm, self.vector_store.as_retriever(), condense_question_prompt)

    def create_stuff_documents_chain(self):
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, qa_prompt)

    def get_answer(self, user_input: str, chat_history: list):
        return self.convo_qa_chain.invoke({"input": user_input, "chat_history": chat_history})

class StreamlitApp:
    def __init__(self, api_key: str, data_dir: str):
        self.api_key = api_key
        self.data_dir = data_dir
        self.processor = DocumentProcessor(data_dir, api_key)
        self.vector_store_manager = VectorStoreManager(self.processor.documents, api_key)
        self.qa_chain = QAChain(self.vector_store_manager.vector_store, api_key)

    def run(self):
        st.title("Document QA System")

        user_input = st.text_input("Ask a question:")
        chat_history = st.session_state.get("chat_history", [])

        if st.button("Submit"):
            if user_input:
                result = self.qa_chain.get_answer(user_input, chat_history)
                st.write(result)
                chat_history.append({"user": user_input, "response": result})
                st.session_state.chat_history = chat_history
            else:
                st.warning("Please enter a question.")

        if st.checkbox("Show chat history"):
            st.write(chat_history)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create and run the Streamlit app
app = StreamlitApp(api_key, "data")
app.run()
