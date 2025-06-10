import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()


# llm 함수 정의
def get_llm(model='gpt-4o-mini'):
    llm = ChatOpenAI(model=model)
    return llm


# DB 함수 정의
def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'quiz-total'

    ## 저장된 인덱스 가져오기
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )

    return database


# QA 함수 정의
def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    database = get_database()

    # prompt = hub.pull('rlm/rag-prompt', api_key=LANGCHAIN_API_KEY)
    prompt = hub.pull("lololll0o/rag-prompt", api_key=LANGCHAIN_API_KEY)

    # llm 모델 지정
    llm = get_llm()

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)


    qa_chain = (
        {
            'context': database.as_retriever() | format_docs,
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain



## RAG[AI Message 함수 정의]
def get_ai_message(user_message):
    qa_chain = get_retrievalQA()
    ai_msg = qa_chain.invoke(user_message)
    return ai_msg
