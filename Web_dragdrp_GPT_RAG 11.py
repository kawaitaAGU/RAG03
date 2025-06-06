import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# OpenAI API Key
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# タイトル
st.title("国家試験問題画像から解析＋類似問題提案")

# CSVの読み込み
try:
    df = pd.read_csv("sample.csv")
    if "問題文" not in df.columns or "a" not in df.columns or "解答" not in df.columns:
        st.error("CSVに必要な列（問題文, a, 解答）が含まれていません。")
        st.stop()
except Exception as e:
    st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
    st.stop()

# ベクトル化準備（問題文）
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df["問題文"].astype(str).tolist())

# 画像アップロード
uploaded_file = st.file_uploader("国家試験問題の画像をアップロードしてください（スクリーンショット等）", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 画像表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 画像バイト変換
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    with st.spinner("GPTに画像を送信して問題文を解析中..."):
        vision_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "以下の画像には国家試験の問題が写っています。問題文、選択肢、正解を抽出してください。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                    ],
                }
            ],
            temperature=0.2
        )
        extracted = vision_response.choices[0].message.content

    st.markdown("### 🔍 解析された問題文と選択肢")
    st.markdown(extracted)

    # 類似検索（問題文のみ使用）
    with st.spinner("類似問題を検索中..."):
        sim_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "user", "content": f"次のテキストから問題文部分だけを取り出してください（日本語）：\n\n{extracted}"}
            ],
            temperature=0.2
        )
        extracted_question = sim_response.choices[0].message.content.strip()

        input_vector = vectorizer.transform([extracted_question])
        similarities = cosine_similarity(input_vector, question_vectors)[0]
        top_indices = similarities.argsort()[::-1][:3]
        similar_rows = df.iloc[top_indices]

    st.markdown("### 🧠 RAGによる類似問題（上位3件）")
    for i, row in similar_rows.iterrows():
        st.markdown(f"**問題：** {row['問題文']}")
        st.markdown(f"a. {row['a']}  b. {row['b']}  c. {row['c']}  d. {row['d']}  e. {row['e']}")
        st.markdown(f"**正解：** {row['解答']}")
        st.markdown("---")

    # 類似問題＋解析結果を再送信し、解説と新規類似問題生成を依頼
    with st.spinner("最終出力を生成中（解説＋新規類似問題の作成）..."):
        final_prompt = f"""
以下は画像から抽出した問題です：
{extracted}

さらに、過去問から抽出された類似問題は以下の通りです：
{similar_rows.to_string(index=False)}

この情報を参考に、
1. 画像の問題の正解と選択肢ごとの説明
2. 新たな類似問題を3題、それぞれ選択肢付き＋解説つき

を日本語で生成してください。
        """

        final_response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3
        )

        final_output = final_response.choices[0].message.content

    st.markdown("### ✅ 正解と詳細な解説")
    st.markdown(final_output)
