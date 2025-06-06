import streamlit as st
import pandas as pd
from PIL import Image
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# GPT-4o モデル使用
MODEL = "gpt-4o-2024-11-20"

# OpenAI API キーの確認
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("歯科医師国家試験 問題解析＆類題生成")

# sample.csvの読み込み（事前に用意されたRAGデータ）
df = pd.read_csv("sample.csv")
questions_list = df["question"].astype(str).tolist()

uploaded_file = st.file_uploader("試験問題画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた問題画像", use_column_width=True)

    # Step 1: Vision GPTで画像から問題文と選択肢を抽出
    with st.spinner("画像から問題文と選択肢を抽出中..."):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "あなたは試験問題のOCRと構造化を行うAIです。"},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "以下は歯科医師国家試験の問題画像です。問題文と選択肢をテキストで抽出し、以下のフォーマットで出力してください。\n\n【問題文】\n...\n【選択肢】\na. ...\nb. ...\nc. ...\nd. ...\ne. ..."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + base64.b64encode(uploaded_file.read()).decode()
                            },
                        },
                    ],
                },
            ],
            temperature=0.2,
        )
        question_text = response.choices[0].message.content
        st.markdown("### 🔍 抽出された問題文と選択肢")
        st.markdown(question_text)

    # Step 2: コサイン類似度によるRAG検索
    with st.spinner("類似問題を検索中..."):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([question_text] + questions_list)
        cos_sim = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        top_k = cos_sim.argsort()[-3:][::-1]
        similar_entries = df.iloc[top_k][["question", "answer", "explanation"]].to_dict(orient="records")

    # Step 3: GPTに解答・解説・類題作成を依頼
    with st.spinner("GPTによる解析と出力を生成中..."):
        prompt = f"""
以下は国家試験問題です。まず正解を選び、選択肢ごとに理由を述べてください。
その後、類似問題を3問作成し、それぞれ正解と解説を加えてください。

【問題】
{question_text}

【参考データ（過去問RAGより）】
{similar_entries}
"""

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "あなたは優秀な歯科医師国家試験の解説者です。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        st.markdown("### ✅ GPTによる解答と解説")
        st.markdown(completion.choices[0].message.content)