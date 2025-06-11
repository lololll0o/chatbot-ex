import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


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

### Statefully manage chat history ###
#  함수 안에 넣으면 안 되는 코드.
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# QA 함수 정의
def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    database = get_database()

    ### Answer question ###
    system_prompt = (
        '''[identity]
        -너는 전세사기피해 법률 전문가야.
        -[context]를 참고하여 사용자의 질문에 답변해.
        -답변 마지막엔 "더 많은 내용은 [너가 참고한 법령(~법 ~조)]에 있습니다."라고 코멘트 달아줘.
        -전세사기피해 법률 관련 이외의 질문에는 "답변할 수 없습니다"라고 출력해.

        [context]
        {context}
        '''
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # llm 모델 지정
    llm = get_llm()

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)


    input_str = RunnableLambda(lambda x: x['input'])

    qa_chain = (
        {
            "context": input_str | database.as_retriever() | format_docs,
            "input": input_str,
            "chat_history" : RunnableLambda(lambda x: x['chat_history']),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_chain_with_history = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return qa_chain_with_history



## RAG[AI Message 함수 정의]
def get_ai_message(user_message, session_id='default'):
    qa_chain = get_retrievalQA()

    ai_msg = qa_chain.invoke(
        {'input' : user_message},
        config={'configurable': {'session_id':session_id}}
    )

    # print(get_session_history(session_id))
    # print('=' * 50 + '\n')

    return ai_msg

