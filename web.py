# 파일명: web.py (Hugging Face API 연동 최종 버전)

# --- 1. 필수 라이브러리 임포트 ---
import streamlit as st
import streamlit.components.v1 as components
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다. (로컬 테스트 시 필요)
load_dotenv()

# rag_logic.py에서 RAG 체인을 가져오는 함수를 임포트합니다.
from rag_logic import get_rag_chain 

# --- 2. 페이지 설정 및 CSS ---
st.set_page_config(page_title="모구챗 - My RAG 챗봇", page_icon="✨", layout="centered")

# (CSS 코드는 원본과 동일)
st.markdown("""
<style>
    /* Noto Sans KR 폰트 로드 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    /* 기본 페이지 스타일 */
    html, body, [class*="st-"] {
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* Streamlit의 메인 콘텐츠 영역 스타일 제거 및 커스텀 */
    .st-emotion-cache-1y4p8pa {
        padding: 0;
    }
    
    /* 전체 앱 컨테이너 */
    .stApp {
        background: linear-gradient(135deg, #F9F5FF 0%, #E2E1FF 100%);
    }

    /* 채팅 컨테이너 (스크롤 영역) */
    .st-emotion-cache-1f1G203 {
        background-color: white;
        border-radius: 1.5rem;
        padding: 1.5rem;
        margin: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        height: 85vh;
        padding-bottom: 5rem;
    }
    
    /* 챗봇(assistant) 메시지 버블 스타일 */
    [data-testid="stChatMessage"][data-testid-role="assistant"] .st-emotion-cache-124el85 {
        background-color: #F0F0F5;
        border-radius: 20px 20px 20px 5px;
        color: #111;
        border: 1px solid #E5E7EB;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* 챗봇(assistant) 아바타 아이콘 스타일 */
    [data-testid="stChatMessage"][data-testid-role="assistant"] .st-emotion-cache-t3u2ir {
        background: linear-gradient(45deg, #7A42E2, #9469F4);
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* 사용자(user) 메시지 버블 스타일 */
    [data-testid="stChatMessage"][data-testid-role="user"] .st-emotion-cache-124el85 {
        background: linear-gradient(45deg, #7A42E2, #9469F4);
        border-radius: 20px 20px 5px 20px;
        color: white;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* FAQ 카드 스타일 */
    .faq-card {
        background-color: rgba(249, 245, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* 추천 질문 버튼 (st.button) 스타일 */
    .stButton>button {
        background-color: #FFFFFF;
        color: #555;
        border: 1px solid #DDD;
        border-radius: 20px;
        padding: 8px 16px;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stButton>button:hover {
        background-color: #F0F0F5;
        color: #7A42E2;
        border-color: #7A42E2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* 채팅 입력창 스타일 */
    .stChatInput {
        background-color: #FFFFFF;
        padding: 1rem;
        border-top: 1px solid #E5E7EB;
    }

    /* 애니메이션 효과 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# --- 3. RAG 챗봇 로직 로드 ---
# NOTE: rag_logic.py 파일이 같은 경로에 있어야 합니다.
rag_chain = get_rag_chain() 

# --- 4. 자동 스크롤 함수 ---
def auto_scroll():
    components.html(
        """<script> window.parent.document.querySelector('.st-emotion-cache-1f1G203').scrollTo(0, 99999); </script>""",
        height=0)

# --- 5. UI 렌더링 함수 ---
def render_welcome_elements():
    # 챗봇 첫인사
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("궁금한 내용을 입력해주시면, 답변을 빠르게 챗봇이 도와드릴게요.")

    # FAQ 카드
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

# rag_chain 로드 성공 시에만 환영 요소 렌더링
if rag_chain:
    if not st.session_state.messages:
        render_welcome_elements()
else:
    # rag_chain 로드 실패 시 (API 키 문제 등)
    with st.chat_message("assistant", avatar="🤖"):
        st.error("챗봇 초기화에 실패했습니다. **HUGGINGFACEHUB_API_TOKEN** 환경 변수를 올바르게 설정했는지 확인해 주세요.")

if st.session_state.messages:
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="✨" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

prompt = st.chat_input("궁금한 내용을 입력하세요...")
if "prompt_from_button" in st.session_state and st.session_state.prompt_from_button:
    prompt = st.session_state.prompt_from_button
    del st.session_state.prompt_from_button

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="✨"):
        if rag_chain:
            response_stream = rag_chain.stream({"question": prompt})
            
            stream_placeholder = st.empty()
            full_response_content = ""
            
            # ⚠️ [중요] LCEL 스트리밍 AttributeError 해결 로직 적용 ⚠️
            for chunk in response_stream:
                # 딕셔너리 형태의 청크일 경우 'answer' 키를 확인합니다.
                if isinstance(chunk, dict) and 'answer' in chunk:
                    content_part = chunk['answer']
                # 이미 문자열로 넘어온 청크일 경우를 대비합니다.
                elif isinstance(chunk, str):
                    content_part = chunk
                else:
                    # 리트리버 출력 등 다른 타입을 무시합니다.
                    continue
                
                full_response_content += content_part
                stream_placeholder.markdown(full_response_content)

            full_response = full_response_content
        else:
            full_response = "죄송합니다, 챗봇을 초기화하는 데 문제가 발생했습니다. **HUGGINGFACEHUB_API_TOKEN** 환경 변수를 확인해 주세요."
            st.write(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    auto_scroll()
    st.rerun()
else:
    auto_scroll()