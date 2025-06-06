import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# OpenAI API クライアント初期化
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# CSVからRAGデータを読み込み
@st.cache_data
def load_rag_data():
    df = pd.read_csv("sample.csv")
    df = df.dropna(subset=["問題文", "解答"])  # 欠損行削除
    return df

df = load_rag_data()

# TF-IDF 類似度検索関数
def retrieve_similar_questions(query, top_k=3):
    corpus = df["問題文"].astype(str).tolist()
    vectorizer = TfidfVectorizer().fit(corpus + [query])
    query_vec = vectorizer.transform([query])
    corpus_vecs = vectorizer.transform(corpus)
    similarities = cosine_similarity(query_vec, corpus_vecs).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

# Streamlit UI
st.title("国家試験問題アシスタント（画像+RAG+GPT）")

uploaded_file = st.file_uploader("国家試験問題の画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた問題", use_column_width=True)

    # 画像 → 問題文抽出 (GPT Vision 1回目)
    with st.spinner("問題文を解析中..."):
        bytes_data = uploaded_file.read()
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは日本の国家試験問題を読み取るアシスタントです。画像から問題文と選択肢を正確に読み取ってください。"},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{bytes_data.decode('latin1')}"}}
                ]}
            ],
            max_tokens=1000
        )
        parsed_question = response.choices[0].message.content
        st.subheader("抽出された問題文")
        st.markdown(parsed_question)

    # 類似問題検索（RAG）
    with st.spinner("類似問題を検索中..."):
        similar_df = retrieve_similar_questions(parsed_question, top_k=3)
        st.subheader("類似問題（RAG検索結果）")
        for i, row in similar_df.iterrows():
            st.markdown(f"**Q{i+1}**: {row['問題文']}")
            st.markdown(f"選択肢: a. {row['a']} / b. {row['b']} / c. {row['c']} / d. {row['d']} / e. {row['e']}")
            st.markdown(f"正解: **{row['解答']}**")

    # GPTによる総合解析（2回目の呼び出し）
    with st.spinner("GPTによる解説と類題生成中..."):
        combined_context = "\n\n".join([
            "【未知の国家試験問題】",
            parsed_question,
            "【類似問題とその選択肢】"
        ] + [
            f"{i+1}. {row['問題文']} (a: {row['a']} b: {row['b']} c: {row['c']} d: {row['d']} e: {row['e']}) 正解: {row['解答']}"
            for i, row in similar_df.iterrows()
        ])

        response2 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "あなたは国家試験の専門家です。与えられた未知の国家試験問題と類似問題に基づき、以下の出力を生成してください：\n1. 未知問題の解答と詳細な解説\n2. 類題を3問とそれぞれの解説"},
                {"role": "user", "content": combined_context}
            ],
            max_tokens=1500
        )
        output = response2.choices[0].message.content
        st.subheader("GPTによる解答・解説と類題")
        st.markdown(output)
