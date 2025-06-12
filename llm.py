import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever, create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

## 환경변수 읽어오기 =====================================================
load_dotenv()

## llm 생성 =========================================================
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

## database 함수 정의 ======================================================
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


## [세션 별 hisotyr] 저장 ========================================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 히스토리 기반 리트리버
def get_history_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        '''
- 너는 대화형 Q&A 시스템의 고도화된 질문 전처리 담당이야.
- 사용자의 질문과 제공된 (chat history)을 참고해서 검색에 가장 적합한 문장으로 사용자의 질문을 명확하게 수정해.
- 'chat history'를 참고해서 현재 질문에 생략된 주어, 목적어 등 맥락상 필요한 정보를 보충하여 질문을 완성해.
- 특히 이전대화에서 언급된 키워드나 명사가 있다면 현재 질문에 자연스럽게 통합시켜.
- 만약 질문이 이미 명확하면 그대로 반환해.
- 질문에 대해 절대 답변하지 말고, 단지 "이해 가능한 독립 질문"으로 고치기만 해.
'''
    )   

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever



def get_prompt():
       ### Answer question ###
    system_prompt = (
    '''[identity]
- 당신은 전세사기피해 법률 전문가입니다.
- [context]를 참고하여 사용자의 질문에 답변하세요.
- 답변에는 해당 조항을 '(XX법 제X조 제X항 제X호, XX법 제X조 제X항 제X호)' 형식으로 문단 마지막에 표시하세요.
- 항목별로 표시해서 답변해주세요.
- 전세사기피해 법률 이외의 질문에는 '답변할 수 없습니다.'로 답하세요.

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
    return qa_prompt

## 전체 체인 구성 =================================================
def build_conversational_chain():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## LLM 모델 지정
    llm = get_llm()

    ## vector store에서 index 정보
    database = get_database()
    retriever = database.as_retriever(search_kwargs={'k': 2})

    ## 히스토리 기반 리트리버 생성
    history_aware_retriever = get_history_retriever(llm, retriever)

    ## QA 프롬프트 생성
    qa_prompt = get_prompt()

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key='answer',
    ).pick('answer')

    return conversational_rag_chain


## [AI Message 함수 정의] ================================================
def stream_ai_message(user_message, session_id='default'):
    qa_chain = build_conversational_chain()

    ai_message = qa_chain.stream(
        {'input': user_message},
        config={'configurable': {'session_id': session_id}},        
    )

    print(f'대화 이력 >> {get_session_history(session_id)} \n🌈\n')
    print('=' * 50 + '\n')

    return ai_message

