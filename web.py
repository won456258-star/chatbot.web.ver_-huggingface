# 파일명: web.py

# --- 1. 필수 라이브러리 임포트 ---
import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv

# rag_logic.py에서 RAG 체인을 가져오는 함수를 임포트
from rag_logic import get_rag_chain

# --- 2. API 키 및 RAG 체인 로드 ---
# Streamlit Secrets에서 API 키를 우선적으로 확인
try:
    # st.secrets에 키가 있는지 확인
    HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except (FileNotFoundError, KeyError):
    # Secrets에 키가 없는 경우 (주로 로컬 개발 환경) .env 파일에서 로드
    print("Secrets not found, loading from .env file.")
    load_dotenv()
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 로드한 API 키를 get_rag_chain 함수에 전달하여 RAG 체인 생성
if HUGGINGFACE_API_KEY:
    rag_chain = get_rag_chain(HUGGINGFACE_API_KEY)
else:
    rag_chain = None # API 키가 없으면 rag_chain을 None으로 설정

# --- 3. 페이지 설정 및 CSS ---
st.set_page_config(page_title="모구챗 - My RAG 챗봇", page_icon="✨", layout="centered")

st.markdown("""
<style>
    /* Noto Sans KR 폰트 로드 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    /* 기본 페이지 스타일 */
    html, body, [class*="st-"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .st-emotion-cache-1y4p8pa { padding: 0; }
    .stApp { background: linear-gradient(135deg, #F9F5FF 0%, #E2E1FF 100%); }
    .st-emotion-cache-1f1G203 {
        background-color: white; border-radius: 1.5rem; padding: 1.5rem; margin: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); border: 1px solid rgba(255, 255, 255, 0.18);
        height: 85vh; padding-bottom: 5rem;
    }
    [data-testid="stChatMessage"][data-testid-role="assistant"] .st-emotion-cache-124el85 {
        background-color: #F0F0F5; border-radius: 20px 20px 20px 5px; color: #111;
        border: 1px solid #E5E7EB; animation: fadeIn 0.5s ease-in-out;
    }
    [data-testid="stChatMessage"][data-testid-role="assistant"] .st-emotion-cache-t3u2ir {
        background: linear-gradient(45deg, #7A42E2, #9469F4); color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stChatMessage"][data-testid-role="user"] .st-emotion-cache-124el85 {
        background: linear-gradient(45deg, #7A42E2, #9469F4); border-radius: 20px 20px 5px 20px;
        color: white; animation: fadeIn 0.5s ease-in-out;
    }
    .faq-card {
        background-color: rgba(249, 245, 255, 0.8); border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.2rem; border-radius: 1rem; margin-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #FFFFFF; color: #555; border: 1px solid #DDD; border-radius: 20px;
        padding: 8px 16px; transition: all 0.2s ease-in-out; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stButton>button:hover {
        background-color: #F0F0F5; color: #7A42E2; border-color: #7A42E2;
        transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stChatInput { background-color: #FFFFFF; padding: 1rem; border-top: 1px solid #E5E7EB; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- 4. 자동 스크롤 함수 ---
def auto_scroll():
    components.html(
        """<script> window.parent.document.querySelector('.st-emotion-cache-1f1G203').scrollTo(0, 99999); </script>""",
        height=0)

# --- 5. UI 렌더링 함수 ---
def render_welcome_elements():
    with st.chat_message("assistant", avatar="✨"):
        st.markdown("궁금한 내용을 입력해주시면, 답변을 빠르게 챗봇이 도와드릴게요.")

    st.markdown('<div class="faq-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 18px; font-weight: 700;"><b>많이 찾는 질문 TOP 3</b></div>', unsafe_allow_html=True)
    faq_items = {
        "모구 수수료 제한은 어떻게 되나요?": "수수료 제한",
        "모구 마감 기한은 며칠까지 가능한가요?": "마감 기한",
        "모구에서 팔면 안되는 물건은 무엇인가요?": "판매 금지 품목"
    }
    for query, text in faq_items.items():
        if st.button(text, key=f"faq_{text}", use_container_width=True):
            st.session_state.prompt_from_button = query
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# --- 6. 메인 애플리케이션 로직 ---
st.title("모구챗 ✨")

if "messages" not in st.session_state:
    st.session_state.messages = []

# RAG 체인 로드 성공 여부에 따라 UI 분기 처리
if not rag_chain:
    st.error("챗봇 초기화에 실패했습니다. API 토큰이 올바르게 설정되었는지 확인해주세요.")
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