
import streamlit as st
import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AskMate | AI FAQ Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
FAQ_FILE = BASE_DIR / "faqs.csv"

# ---------------- CSS ----------------
st.markdown("""

<style>
.stApp {
    background: #0B0B0D;
    color: #F5F5F5;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background: #141416;
    border-right: 1px solid #303035;
}
[data-testid="stSidebar"] * {
    color: #F5F5F5;
}
.hero {
    padding: 30px;
    border: 1px solid #70401F;
    border-radius: 20px;
    background: linear-gradient(120deg, #191512, #302016, #171313);
    margin-bottom: 22px;
}
.hero h1 {
    color: #FF8A35 !important;
    font-size: 38px;
    margin: 0;
}
.hero p {
    color: #E8DCD3 !important;
    font-size: 16px;
    line-height: 1.8;
}
.section-card, .metric-card {
    background: #171719;
    border: 1px solid #353539;
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}
.metric-label {
    color: #B8B8C0 !important;
    font-size: 13px;
}
.metric-value {
    color: #FF8A35 !important;
    font-size: 25px;
    font-weight: 700;
    margin-top: 5px;
}
.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px;
    border: 1px solid #70401F;
    background: #FF7A1A;
    color: #111111 !important;
    min-height: 42px;
    font-weight: 600;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #FF9A4D;
    border-color: #FFB276;
    color: #111111 !important;
}
/* Chat cards */
[data-testid="stChatMessage"] {
    background: #171719 !important;
    border: 1px solid #38383D !important;
    border-radius: 15px;
    padding: 16px;
    margin-bottom: 12px;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #F5F5F5 !important;
    -webkit-text-fill-color: #F5F5F5 !important;
    line-height: 1.8 !important;
    overflow-wrap: anywhere;
}
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #FF9A4D !important;
    -webkit-text-fill-color: #FF9A4D !important;
}
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p {
    color: #B8B8C0 !important;
    -webkit-text-fill-color: #B8B8C0 !important;
}
/* Chat input: readable typed text */
[data-testid="stChatInput"] {
    background: #19191D !important;
    border: 1px solid #70401F !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    
[data-testid="stChatInput"] textarea {
    background-color: #19191D !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FF8A35 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #B8B8C0 !important;
    -webkit-text-fill-color: #B8B8C0 !important;
    opacity: 1 !important;
}

}
[data-testid="stChatInput"] textarea::placeholder {
    color: #B8B8C0 !important;
    -webkit-text-fill-color: #B8B8C0 !important;
    opacity: 1 !important;
}
/* Other text inputs */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background: #19191D !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
h2, h3 {
    color: #FF8A35 !important;
}
small, [data-testid="stCaptionContainer"] {
    color: #B8B8C0;
}
</style>

""", unsafe_allow_html=True)


# --------------- FAQ DATA ---------------
@st.cache_data
def load_faqs(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    if not {"Question", "Answer"}.issubset(data.columns):
        raise ValueError(
            "faqs.csv needs Question and Answer columns."
        )

    data = data[["Question", "Answer"]].dropna().copy()
    data["Question"] = data["Question"].astype(str).str.strip()
    data["Answer"] = data["Answer"].astype(str).str.strip()

    data = data[
        (data["Question"] != "") &
        (data["Answer"] != "")
    ]
    return data.drop_duplicates(
        subset=["Question"]
    ).reset_index(drop=True)


try:
    df = load_faqs(str(FAQ_FILE))
except Exception as error:
    st.error(f"Could not read faqs.csv: {error}")
    st.stop()

if df.empty:
    st.error("Your FAQ dataset is empty. Add questions and answers.")
    st.stop()


# --------------- NLP ENGINE ---------------
@st.cache_resource
def build_search_engine(question_list):
    # Word features find meaningful terms and short phrases.
    word_model = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    # Character features help with spelling variations and
    # questions that use slightly different word forms.
    char_model = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )

    word_matrix = word_model.fit_transform(question_list)
    char_matrix = char_model.fit_transform(question_list)

    return word_model, char_model, word_matrix, char_matrix


word_model, char_model, word_matrix, char_matrix = (
    build_search_engine(tuple(df["Question"].tolist()))
)


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def find_answer(user_question):
    query = clean_text(user_question)

    if not query:
        return None

    # Direct question match
    for index, question in enumerate(df["Question"]):
        if query == clean_text(question):
            return {
                "answer": df.iloc[index]["Answer"],
                "question": question,
                "score": 1.0,
                "found": True,
            }

    word_query = word_model.transform([user_question])
    char_query = char_model.transform([user_question])

    word_scores = cosine_similarity(
        word_query, word_matrix
    )[0]
    char_scores = cosine_similarity(
        char_query, char_matrix
    )[0]

    # Combining both signals is more robust than using only words.
    scores = 0.65 * word_scores + 0.35 * char_scores
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])

    # This is a similarity score, not a probability.
    if best_score < 0.22:
        return {
            "answer": (
                "I couldn't find a reliable answer in my FAQ "
                "knowledge base. Try a suggested question, "
                "or rephrase your question. You can also add "
                "more FAQs to faqs.csv."
            ),
            "question": None,
            "score": best_score,
            "found": False,
        }

    return {
        "answer": df.iloc[best_index]["Answer"],
        "question": df.iloc[best_index]["Question"],
        "score": best_score,
        "found": True,
    }


# --------------- SESSION STATE ---------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --------------- SIDEBAR ---------------
with st.sidebar:
    st.markdown("# 🤖 AskMate")
    st.caption("AI FAQ Assistant")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠  Home")
    st.markdown("💬  Chat Assistant")
    st.markdown("📚  FAQ Knowledge Base")
    st.markdown("ℹ️  About Project")

    st.divider()
    st.markdown("### Project Details")
    st.write("**Language:** Python")
    st.write("**Interface:** Streamlit")
    st.write("**NLP:** TF-IDF")
    st.write("**Matching:** Cosine similarity")

    st.divider()
    st.markdown("### Smart Answers")
    st.caption(
        "Find answers from your FAQ dataset. "
        "Unlisted topics may need additional FAQs."
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# --------------- HEADER ---------------
st.markdown("""
<div class="hero">
    <h1>👋 Hello! I'm AskMate</h1>
    <p>Your AI-powered FAQ assistant</p>
    <p>
        Ask your question in your own words.
        I'll search the knowledge base for the closest answer.
    </p>
</div>
""", unsafe_allow_html=True)


# --------------- METRICS ---------------
user_count = sum(
    1 for message in st.session_state.messages
    if message["role"] == "user"
)

c1, c2, c3 = st.columns(3)

for col, label, value in [
    (c1, "📚 Knowledge Base", f"{len(df)} FAQs"),
    (c2, "💬 Questions Asked", str(user_count)),
    (c3, "🧠 Search Engine", "NLP + TF-IDF"),
]:
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.write("")

# --------------- SUGGESTED QUESTIONS ---------------
st.markdown("### ✨ Suggested questions")
st.caption("Select a question to try the chatbot.")

suggestions = df["Question"].head(6).tolist()

for start in range(0, len(suggestions), 3):
    row = st.columns(3)
    for offset, question in enumerate(
        suggestions[start:start + 3]
    ):
        with row[offset]:
            if st.button(
                "💬 " + question,
                key=f"suggestion_{start + offset}",
                use_container_width=True,
            ):
                st.session_state.pending_question = question

st.write("")
st.markdown("### 💬 Chat with AskMate")

# --------------- CHAT HISTORY ---------------
if not st.session_state.messages:
    st.markdown(
        '<div class="section-card">'
        '<h3>Welcome to AskMate!</h3>'
        '<p>Start by selecting a suggested question above, '
        'or type your own question below.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("matched_question"):
            st.caption(
                "Matched FAQ: " + message["matched_question"]
            )

        if message.get("score") is not None:
            st.caption(
                f"Similarity score: {message['score']:.2f} "
                "(not a probability)"
            )


# --------------- USER INPUT ---------------
typed_question = st.chat_input(
    "Type your question here..."
)

question_to_process = (
    st.session_state.pending_question or typed_question
)
st.session_state.pending_question = None

if question_to_process:
    st.session_state.messages.append({
        "role": "user",
        "content": question_to_process,
    })

    result = find_answer(question_to_process)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "matched_question": result["question"],
        "score": result["score"],
    })

    st.rerun()


# --------------- FOOTER ---------------
st.divider()
st.caption(
    "AskMate • Internship Project • Python | Streamlit | NLP"
)
st.caption(
    "Answers are based on the supplied FAQ dataset. "
    "Similarity scores are not calibrated probabilities."
)
