import streamlit as st
from PIL import Image
import pandas as pd
from io import BytesIO
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API Key 読み込み
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。StreamlitのSecretsに設定してください。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# CSVデータ読み込み
df = pd.read_csv("sample.csv")
if "問題文" not in df.columns or "a" not in df.columns or "解答" not in df.columns:
    st.error("CSVに必要な列（問題文, a, 解答）がありません")
    st.stop()

# 類似問題検索関数
def retrieve_similar_questions(query, top_k=3):
    vectorizer = TfidfVectorizer().fit(df["問題文"])
    tfidf_matrix = vectorizer.transform(df["問題文"])
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    return df.iloc[top_indices]

# Streamlit UI
st.title("画像ベース歯科国家試験問題解析＋RAGシステム")
uploaded_file = st.file_uploader("国家試験問題の画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="アップロードされた画像", use_column_width=True)

    # 画像をGPT-4oに送信し問題解析
    image_bytes = uploaded_file.read()
    image_analysis = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "あなたは国家試験問題に特化したOCRと問題解析を行うアシスタントです。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "この画像に含まれる国家試験問題の問題文、選択肢（a〜e）、正解を抽出してください。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_bytes.hex()}", "detail": "high"}}
                ],
            },
        ],
        temperature=0.3,
        max_tokens=2000
    )

    parsed_text = image_analysis.choices[0].message.content
    st.markdown("### 🧾 解析結果")
    st.markdown(parsed_text)

    # 類似問題をRAG検索
    similar_df = retrieve_similar_questions(parsed_text, top_k=3)

    st.markdown("### 🔍 類似問題 (RAG結果)")
    for i, row in similar_df.iterrows():
        st.markdown(f"**問題**: {row['問題文']}")
        st.markdown(f"a: {row['a']}　b: {row['b']}　c: {row['c']}　d: {row['d']}　e: {row['e']}")
        st.markdown(f"**正解**: {row['解答']}")
        st.markdown("---")

    # GPTに再送信して解答＆類題生成
    rag_context = "\n\n".join(
        f"問題: {row['問題文']}\na: {row['a']}\nb: {row['b']}\nc: {row['c']}\nd: {row['d']}\ne: {row['e']}\n正解: {row['解答']}"
        for _, row in similar_df.iterrows()
    )

    final_response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "あなたは国家試験問題の出題意図を深く理解し、解説と類題を生成するアシスタントです。"},
            {"role": "user", "content": f"""
以下は解析された未知の問題です。

{parsed_text}

以下はRAG（過去問題の類似問題）です：

{rag_context}

この情報をもとに、1. 問題文と各選択肢の解説、2. 類似問題を3問、それぞれの選択肢ごとの説明と正解を生成してください。
"""}
        ],
        temperature=0.3,
        max_tokens=3000
    )

    st.markdown("### ✅ 解答・解説と類題")
    st.markdown(final_response.choices[0].message.content)
