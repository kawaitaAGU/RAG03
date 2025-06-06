import streamlit as st
import pandas as pd
from PIL import Image
import base64
import io
from openai import OpenAI
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI APIキーの取得
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# データ読み込み
csv_path = Path("sample.csv")
df = pd.read_csv(csv_path)

# カラムの確認
required_columns = ["設問", "選択肢a", "選択肢b", "選択肢c", "選択肢d", "選択肢e", "正解"]
if not all(col in df.columns for col in required_columns):
    st.error("sample.csv のカラム名が不正です。以下のカラムが必要です: " + ", ".join(required_columns))
    st.stop()

# TF-IDF準備
corpus = df["設問"].astype(str).tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def retrieve_similar_questions(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

# Streamlit UI
st.title("歯科医師国家試験問題 AI解析・RAG解析・類題アプリ")

uploaded_file = st.file_uploader("国家試験問題の画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    with st.spinner("GPT-4oが画像から問題文を読み取り中..."):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        base64_image = base64.b64encode(image_bytes.read()).decode()

        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは国家試験画像から問題を抽出するOCRの専門家です。問題文と選択肢を丁寧に抽出してください。"},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ],
            max_tokens=1000
        )

        extracted_question = response.choices[0].message.content.strip()
        st.markdown("### 抽出された問題文")
        st.markdown(extracted_question)

    with st.spinner("類似問題を検索中..."):
        similar_df = retrieve_similar_questions(extracted_question)
        st.markdown("### 類似問題（RAGから検索）")
        for i, row in similar_df.iterrows():
            st.markdown(f"**Q{i+1}: {row['設問']}**")
            st.markdown(f"a. {row['選択肢a']} 　b. {row['選択肢b']} 　c. {row['選択肢c']} 　d. {row['選択肢d']} 　e. {row['選択肢e']}")
            st.markdown(f"**正解:** {row['正解']}")

    with st.spinner("GPT-4oが最終解説と類題を生成中..."):
        rag_text = "\n\n".join(
            f"{i+1}. {row['設問']}\na. {row['選択肢a']}\nb. {row['選択肢b']}\nc. {row['選択肢c']}\nd. {row['選択肢d']}\ne. {row['選択肢e']}\n正解: {row['正解']}"
            for i, row in similar_df.iterrows()
        )

        final_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは歯科国家試験の教育専門家です。未知の問題と類似問題の内容から、正解・理由の解説、さらに類題を3つ作成してください。"},
                {"role": "user", "content": f"【未知の問題】\n{extracted_question}\n\n【類似問題】\n{rag_text}"}
            ],
            max_tokens=2000
        )

        st.markdown("### GPTによる正解・解説・類題生成")
        st.markdown(final_response.choices[0].message.content.strip())
