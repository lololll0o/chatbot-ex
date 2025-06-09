import streamlit as st

st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤖')
st.title("전세사기피해 상담 챗봇🤖")


if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message["content"])

# prompt 창(채팅창)
placeholder = "이곳에서는 전세사기 피해와 관련된 질문만 가능합니다."
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        # 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

with st.chat_message('ai'):
    st.write('ai_msg')
st.session_state.message_list.append({'role':'ai', 'content':'ai_msg'})

print(f'after: {st.session_state.message_list}')