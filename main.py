
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class MobilePackagesChatbot:
    def __init__(self):
        self.data = None
        self.docsearch = None
        self.qa_model = None
        self.chat_history = []

    def load_data(self, file_path):
        try:
            loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
            return loader.load()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def compute_embeddings(self, data, chunk_size=500, chunk_overlap=20, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            text_chunks = text_splitter.split_documents(data)

            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            docsearch = FAISS.from_documents(text_chunks, embeddings)
            return docsearch
        except Exception as e:
            st.error(f"Error computing embeddings: {e}")
            return None

    def init_llm_model(self, model_path, retriever):
        try:
            llm = CTransformers(model=model_path, model_type="llama")
            qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever.as_retriever())
            return qa
        except Exception as e:
            st.error(f"Error initializing the LLM model: {e}")
            return None

    def run(self):
        st.title('Customized Mobile Packages Chatbot')
        st.markdown("Welcome to the Mobile Packages Chatbot! Ask any question related to mobile packages, and I'll do my best to assist you.")
    
        self.data = self.load_data("data/mobile_packages.csv")
        if self.data is None:
            return

        self.docsearch = self.compute_embeddings(self.data)
        if self.docsearch is None:
            return

        self.qa_model = self.init_llm_model("models/llama-2-7b-chat.ggmlv3.q8_0.bin", self.docsearch)
        if self.qa_model is None:
            return
    
        st.markdown("### Ask about mobile packages, plans, features, or any related queries.")
        user_input = st.text_input("Ask something:")
        if st.button('Ask the Chatbot'):
            if user_input:
                response_dict = self.qa_model({"question": user_input, "chat_history": self.chat_history})
                response = response_dict.get('answer', 'Sorry, I could not generate a response.')
                self.chat_history.append({"user": user_input, "bot": response})
                for chat in self.chat_history:
                    st.text(f"User: {chat['user']}")
                    st.text(f"Bot: {chat['bot']}")

if __name__ == "__main__":
    chatbot = MobilePackagesChatbot()
    chatbot.run()
