import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate

ex_prompt = PromptTemplate.from_template("Question : {input}\n{answer}")


load_dotenv()

## LLM 생성 =======================================================================
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

## Embedding 설정 + Vector Store Index 가져오기 ==========================#==========
def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'quiz-total'

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )

    return database

### 세션별 히스토리 저장 ==============================================================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## 히스토리 기반 리트리버 =============================================================
def get_history_retriever(llm, retriever):
    # 질문을 자세히 받아들일 수 있도록 도와주는 명령프롬프트
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
    # 질문 재구성 프롬프트, 사용자의 질문에 답변 x
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

def get_qa_prompt():
    system_prompt = (
    '''[identity]
-너는 전세사기피해 법률 전문가야.
-[context]를 참고하여 사용자의 질문에 답변해.
-답변 마지막엔 "더 많은 내용은 (너가 참고한 법령 ex) ~법 ~조)에 있습니다."라고 코멘트 달아줘.
-전세사기피해 법률 관련 이외의 질문에는 "답변할 수 없습니다"라고 출력해.

[Context]
{context} 
'''
    )
    # few - shot
    from langchain_core.prompts import PromptTemplate
    from langchain_core.prompts import FewShotPromptTemplate
    from config import answer_ex

    ex_prompt = PromptTemplate.from_template("질문 : {input}\n\n 답변 :{answer}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_ex, # 질문/답변 예시들 (type(전체 : list[], 각각문서 : dict{}))
        example_prompt=ex_prompt,
        prefix="다음 질문에 답변하시오. : ",
        suffix="Question: {input}",
        input_variables=["input"],
    )

    formated_few_shot_prompt = few_shot_prompt.format(input='{input}')
    


    # 사용자의 질문에 과거의 기록을 보고 답변 (실질 프롬프트)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ('assistant', formated_few_shot_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return qa_prompt

## 전체 chain 구성 ===============================================================
def build_conversational_chain(): 
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    llm = get_llm()

    ## vector store에서 index 정보
    database = get_database()
    retriever = database.as_retriever(search_kwargs={"k": 2})

    history_aware_retriever = get_history_retriever(llm, retriever)

    qa_prompt = get_qa_prompt()
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


## AI Message ===========================================================================
def stream_ai_message(user_message, session_id='default'):
    qa_chain = build_conversational_chain()

    ai_message = qa_chain.stream(
        {"input": user_message}, 
        config={"configurable": {"session_id": session_id}}
    )

    print(f'대화 이력 >> { get_session_history(session_id)}\n\n')
    print('='*50+'\n')
    print(f'[stream_ai_message 함수 내 출력]session_id >>{session_id}')
    return ai_message