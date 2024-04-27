import streamlit as st
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_core.messages import SystemMessage  # type: ignore
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain_text_splitters import NLTKTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
from langchain_core.runnables import RunnablePassthrough # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore

f = open('keys/gemini_key.txt')
GOOGLE_API_KEY = f.read()

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")


# Define Streamlit title and header
st.title("👩🏻‍💻 RAG App -Leave No Context Behind paper 👨🏻‍💻")
st.header("I'm your AI Assistant 🤖. Feel free to ask me any doubts regarding it.")

# getting the input from  user input
user_input = st.text_area("Enter your Question or doubts.")


if st.button("Generate"):
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""Hi, I am Rager, your friendly AI assistant.
        I'm here to provide you with insightful answers based on your questions and context."""),
        HumanMessagePromptTemplate.from_template("""Let's unreveal some valuable insight together!
        Context:
        {context}
        
        Your Question: 
        {question}
        
        My Response: """)
    ])

    chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-pro-latest")

    
    output_parser = StrOutputParser()

    # now defining the pdf loader
    loader = PyPDFLoader("./no_context_paper.pdf")
    pages = loader.load()

    # split PDF documents into chunks
    text_splitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=80)
    chunks = text_splitter.split_documents(pages)

    # create embeddings for the chunks
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
    db = Chroma.from_documents(chunks, embedding_model, persist_directory="./my_chroma_db_")

    # define retriever and format docs function
    retriever = db.as_retriever(search_kwargs={"k": 8})
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # define RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | chat_template
        | chat_model
        | output_parser
    )

    # now invoke RAG chain to generate answer
    response = rag_chain.invoke(user_input)
    st.subheader("Result:")
    st.write(response)   