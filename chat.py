import streamlit as st
import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone



st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤖')
st.title("전세사기피해 상담 챗봇🤖")


if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message["content"])

## RAG[AI Message 함수 정의]
def get_ai_message(user_message):
    ## 환경변수 읽어오기 ############################################
    load_dotenv()
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


    ## 벡터 스토어(데이터베이스)에서 인덱스 가져오기 ###############
    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'quiz-total'


    ## 저장된 인덱스 가져오기
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )


    ## RetrievalQA ##################################################
    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull('rlm/rag-prompt')


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


    ai_msg = qa_chain.invoke(user_message)
    return ai_msg



# prompt 창(채팅창)
placeholder = "이곳에서는 전세사기 피해와 관련된 질문만 가능합니다."
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        # 사용자 msg 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    ai_msg = get_ai_message(user_question)

    with st.chat_message('ai'):
        #AI msg 화면 출력
        st.write(ai_msg)
    st.session_state.message_list.append({'role':'ai', 'content':ai_msg})

print(f'after: {st.session_state.message_list}')