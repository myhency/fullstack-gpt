import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOllama, ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.llms import OpenAI

st.set_page_config(page_title="DocumentGPT", page_icon="📜")

MODEL="gpt-3.5-turbo-0125"
EMBEDDING_MODEL="gpt-3.5-turbo-0125"

@st.cache_resource
def init_llm(chat_callback: bool):
    if chat_callback:

        class ChatCallbackHandler(BaseCallbackHandler):
            def __init__(self, *args, **kwargs):
                self.tokens = ""

            def on_llm_start(self, *args, **kwargs):
                self.messagebox = st.empty()
                self.tokens = ""

            def on_llm_end(self, *args, **kwargs):
                save_message(role="ai", message=self.tokens)

            def on_llm_new_token(self, token, *args, **kwargs):
                self.tokens += token
                with self.messagebox:
                    st.write(self.tokens)

        callbacks = [ChatCallbackHandler()]
    else:
        callbacks = []

    return ChatOpenAI(
        model=MODEL,
        temperature=0.1,
        streaming=True,
        callbacks=callbacks,
    )

llm_for_chat = init_llm(chat_callback=True)
llm_for_memory = init_llm(chat_callback=False)

@st.cache_resource
def init_memory(_llm):
    return ConversationSummaryBufferMemory(
        llm=_llm, max_token_limit=60, return_messages=True, memory_key="chat_history"
    )

memory = init_memory(llm_for_memory)

def save_message(role, message):
    st.session_state["messages"].append({"role": role, "message": message})

def show_message(role, message, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        save_message(role, message)

@st.cache_data(show_spinner="File is uploaded...")
def handle_file(file):
    # Save the uploaded file
    upload_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(upload_content)
    
    # Load the file and split the text
    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=30
    )
    load_docs = loader.load()
    split_docs = splitter.split_documents(load_docs)
    
    # Convert each split document to descriptive text using LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")  # LLM for descriptive conversion
    descriptive_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 텍스트를 서술형으로 한국어로 다시 작성해 주세요. 단락에 제목을 달지말고 모든 내용을 서술형으로 작성해주세요. Please rewrite the following text in Korean as a narrative. Please do not add titles to paragraphs and write all content in descriptive form."),
        ("human", "{text}")
    ])
    chain = LLMChain(prompt=descriptive_prompt, llm=llm)
    
    descriptive_texts = []
    for doc in split_docs:
        descriptive_text = chain.run({"text": doc.page_content})
        descriptive_texts.append(descriptive_text)
    
    descriptive_texts
    
    # Embed the descriptive texts
    embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=OpenAIEmbeddings(),
        document_embedding_cache=LocalFileStore("./.cache/embeddings/"),
    )
    embedding_in_vectorstore = FAISS.from_texts(descriptive_texts, embedder)

    return embedding_in_vectorstore.as_retriever()

def load_memory(input):
    return memory.load_memory_variables({})["chat_history"]

def show_prompt(inputs):
    print(inputs)
    return inputs

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context and the conversation history. If you don't know the answer just say you don't know. DON'T make anything up.
            답변은 반드시 한국어로 해.
            
            Context: {context}
            """,
        ),
        ("human", "*****************************************************************"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "ai", "message": "I'm ready! Ask anything about your file."}
    ]

# Main app implementation

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your file on the sidebar!
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt, .pdf, .md or .docx files!", ["txt", "pdf", "docx", "html", "md"]
    )

if file:
    retriever = handle_file(file)
    for msg in st.session_state["messages"]:
        show_message(role=msg["role"], message=msg["message"], save=False)
    new_message = st.chat_input("Send your message")
    if new_message:
        show_message(role="human", message=new_message)

        chain = (
            {
                "context": retriever
                | RunnableLambda(
                    lambda docs: "\n\n".join(doc.page_content for doc in docs)
                ),
                "question": RunnablePassthrough(),
                "chat_history": load_memory,
            }
            | RunnableLambda(show_prompt)
            | prompt
            | llm_for_chat
        )
        with st.chat_message("ai"):
            response = chain.invoke(new_message).content
            memory.save_context({"input": new_message}, {"output": response})
else:
    st.session_state["messages"] = [
        {"role": "ai", "message": "I'm ready! Ask anything about your file."}
    ]
    memory.clear()
    memory.save_context(
        {"input": "Here is my file"},
        {"output": "I'm ready! Ask anything about your file."},
    )