# 파일명: web.py

# --- 1. 필수 라이브러리 임포트 ---
import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv

# ✅ 1. 페이지 설정을 가장 먼저 실행합니다.
st.set_page_config(page_title="모구챗 - My RAG 챗봇", page_icon="✨", layout="centered")

# ✅ 2. 페이지 설정 이후에 다른 모듈을 임포트합니다.
from rag_logic import get_rag_chain, RagChainInitializationError

# --- 2. API 키 및 RAG 체인 로드 ---
rag_chain = None  # 기본값을 None으로 설정
try:
    # Streamlit Secrets 또는 .env에서 API 키 로드
    try:
        HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except (FileNotFoundError, KeyError):
        load_dotenv()
        HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # ✅ 3. rag_logic의 오류를 여기서 직접 처리합니다.
    if HUGGINGFACE_API_KEY:
        rag_chain = get_rag_chain(HUGGINGFACE_API_KEY)
    else:
        st.error("Hugging Face API 토큰을 찾을 수 없습니다. Secrets 또는 .env 파일을 확인해주세요.")

except RagChainInitializationError as e:
    # rag_logic에서 보낸 예외를 받아서 UI에 에러 메시지를 표시합니다.
    st.error(e)

# --- 3. CSS ---
st.markdown("""
<style>
    /* CSS 내용은 이전과 동일 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Noto Sans KR', sans-serif; }
    .st-emotion-cache-1y4p8pa { padding: 0; }
    /* ... (나머지 CSS 생략) ... */
</style>
""", unsafe_allow_html=True)


# --- 4. 자동 스크롤 함수 ---
def auto_scroll():
    components.html(
        """<script> window.parent.document.querySelector('.st-emotion-cache-1f1G203').scrollTo(0, 99999); </script>""",
        height=0)

# --- 5. UI 렌더링 함수 ---
def render_welcome_elements():
    # ... (이전과 동일) ...
    with st.chat_message("assistant", avatar="✨"):
        st.markdown("궁금한 내용을 입력해주시면, 답변을 빠르게 챗봇이 도와드릴게요.")
    # ... (이하 생략) ...

# --- 6. 메인 애플리케이션 로직 ---
st.title("모구챗 ✨")

if "messages" not in st.session_state:
    st.session_state.messages = []

# RAG 체인 로드 성공 여부에 따라 UI 분기 처리
if not rag_chain:
    # 에러 메시지는 이미 위 try-except 블록에서 표시했으므로 여기서는 아무것도 안 함
    pass
else:
    # 이전 대화 내용 표시
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])
    else:
        render_welcome_elements()

    # 사용자 입력 처리
    prompt = st.chat_input("궁금한 내용을 입력하세요...")
    # ... (이하 로직은 이전과 동일) ...
    if "prompt_from_button" in st.session_state and st.session_state.prompt_from_button:
        prompt = st.session_state.prompt_from_button
        del st.session_state.prompt_from_button

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("답변을 생성하고 있어요... 잠시만 기다려주세요."):
                full_response = rag_chain.invoke({"question": prompt})
            st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# 항상 자동 스크롤 실행
auto_scroll()