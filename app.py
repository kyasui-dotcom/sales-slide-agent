import os
import json
import io
import requests
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """あなたは、B2B商材の市場分析と営業戦略の立案を行うプロフェッショナルな営業コンサルタントです。

# 思考プロセス
1. 市場の「不都合な真実」を指摘：クライアントが狙っている市場の難所（既得権益、予算不足等）を冷静に分析し、あえて「そこは厳しい」と伝える。
2. ブルーオーシャン（新用途）の提示：商品の特性を活かし、別の「深い悩み（集客、売上、ブランド）」を持つターゲットへスライドさせる。
3. 時間軸を捉えた営業フロー：単なるアポ取りではなく、顧客の「予算編成時期」や「意思決定サイクル」を逆算したプロセスを構築する。

# 出力ルール
以下のJSON配列形式で出力してください。各要素が1枚のスライドになります。
マークダウン記法を使用してください（箇条書き、太字、表など）。

```json
[
  {
    "title": "スライドタイトル",
    "content": "スライドの本文（マークダウン形式）",
    "type": "cover|analysis|proposal|flow|pricing|summary"
  }
]
```

# スライド構成（必ずこの順序で6枚以上）
1. **表紙** (type: "cover") - 提案タイトルと対象商材名
2. **市場環境分析** (type: "analysis") - 市場の不都合な真実、競合状況、参入障壁（複数枚可）
3. **新ターゲット提案** (type: "proposal") - ブルーオーシャン戦略、新しいターゲット層、なぜそこが狙い目か（複数枚可）
4. **営業フロー** (type: "flow") - 時間軸を捉えた具体的な営業プロセス、予算編成時期の逆算（複数枚可）
5. **支援範囲と成果報酬** (type: "pricing") - 具体的な支援内容と成果報酬体系
6. **まとめ** (type: "summary") - 要点整理と次のアクション

トーンは客観的で冷静、かつ解決策に対しては情熱的であること。
表や箇条書きを多用し、そのまま提案資料として使える品質にすること。
JSON配列のみを出力し、それ以外のテキストは含めないこと。"""


def extract_text_from_url(url):
    """URLからテキストを抽出"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # 長すぎる場合は切り詰め
    return text[:8000]


def extract_text_from_pdf(file_storage):
    """アップロードされたPDFからテキストを抽出"""
    reader = PdfReader(io.BytesIO(file_storage.read()))
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    text = "\n".join(text_parts)
    return text[:8000]


def generate_slides(product_info):
    """OpenAI APIで提案資料スライドを生成"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"以下の商品・サービス情報をもとに、営業提案資料のスライドを作成してください。\n\n{product_info}",
            },
        ],
        temperature=0.7,
        max_tokens=4000,
    )
    content = response.choices[0].message.content.strip()
    # JSON部分を抽出（```json ... ``` で囲まれている場合に対応）
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return json.loads(content)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    input_type = request.form.get("input_type", "text")

    try:
        if input_type == "url":
            url = request.form.get("url", "").strip()
            if not url:
                return jsonify({"error": "URLを入力してください"}), 400
            product_info = extract_text_from_url(url)

        elif input_type == "pdf":
            if "pdf_file" not in request.files:
                return jsonify({"error": "PDFファイルを選択してください"}), 400
            pdf_file = request.files["pdf_file"]
            if pdf_file.filename == "":
                return jsonify({"error": "PDFファイルを選択してください"}), 400
            product_info = extract_text_from_pdf(pdf_file)

        elif input_type == "text":
            product_info = request.form.get("text_input", "").strip()
            if not product_info:
                return jsonify({"error": "商品情報を入力してください"}), 400

        else:
            return jsonify({"error": "無効な入力タイプです"}), 400

        if len(product_info.strip()) < 10:
            return jsonify({"error": "商品情報が短すぎます。もう少し詳しい情報を入力してください"}), 400

        slides = generate_slides(product_info)
        return jsonify({"slides": slides})

    except requests.RequestException as e:
        return jsonify({"error": f"URLの取得に失敗しました: {str(e)}"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "AIからの応答の解析に失敗しました。もう一度お試しください"}), 500
    except Exception as e:
        return jsonify({"error": f"エラーが発生しました: {str(e)}"}), 500


@app.route("/slides")
def slides():
    return render_template("slides.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
