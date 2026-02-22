import os
import json
import io
import requests
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# ユーザーが編集できるプロンプト部分（役割・思考プロセス・トーン）
DEFAULT_ROLE_PROMPT = """あなたは、B2B商材の市場分析と営業戦略の立案を行うプロフェッショナルな営業コンサルタントです。

# 思考プロセス
1. 市場の「不都合な真実」を指摘：クライアントが狙っている市場の難所（既得権益、予算不足等）を冷静に分析し、あえて「そこは厳しい」と伝える。
2. ブルーオーシャン（新用途）の提示：商品の特性を活かし、別の「深い悩み（集客、売上、ブランド）」を持つターゲットへスライドさせる。
3. 時間軸を捉えた営業フロー：単なるアポ取りではなく、顧客の「予算編成時期」や「意思決定サイクル」を逆算したプロセスを構築する。

# トーン
- 客観的で冷静、かつ解決策に対しては情熱的であること。
- 表や箇条書きを多用し、そのまま提案資料として使える品質にすること。
- 具体的な数値や事例を含め、説得力のある内容にすること。"""

# 商品認識・市場分析の確認用プロンプト
ANALYZE_PROMPT = """あなたは、B2B商材の市場分析を行うプロフェッショナルな営業コンサルタントです。

以下の商品・サービス情報を読み取り、あなたの理解をJSON形式で整理してください。
このあと、この理解をベースに営業提案資料を作成します。

# 出力形式（必ずこのJSON形式で出力）
```json
{
  "product_name": "商品・サービス名",
  "product_summary": "商品・サービスの概要（2-3文）",
  "strengths": ["強み1", "強み2", "強み3"],
  "target_market": "現在想定されるターゲット市場",
  "market_size": "市場規模の推定（わかる範囲で）",
  "competitors": ["主な競合1", "主な競合2", "主な競合3"],
  "market_challenges": ["市場の課題1", "市場の課題2"],
  "price_range": "価格帯（わかる範囲で）",
  "blue_ocean_hint": "ブルーオーシャンの可能性（一言で）"
}
```

JSON以外のテキストは含めないこと。"""

# システム固定部分（JSON出力ルール）- ユーザーからは編集不可
SYSTEM_OUTPUT_RULES = """
# 出力ルール（※この部分はシステム固定です）
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

# スライド構成（必ずこの順序で8枚以上）
1. **表紙** (type: "cover") - 提案タイトルと対象商材名
2. **市場環境分析** (type: "analysis") - 市場の不都合な真実、競合状況、参入障壁。表やデータを交えて複数枚で詳しく。
3. **新ターゲット提案** (type: "proposal") - ブルーオーシャン戦略、新しいターゲット層、なぜそこが狙い目か。具体的な業界・企業規模・課題を明記。複数枚で詳しく。
4. **営業フロー** (type: "flow") - 時間軸を捉えた具体的な営業プロセス、予算編成時期の逆算。月別・週別のアクション表を含める。複数枚で詳しく。
5. **支援範囲と成果報酬** (type: "pricing") - 具体的な支援内容と成果報酬体系。表形式で明確に。
6. **まとめ** (type: "summary") - 要点整理と次の具体的アクション

各スライドの内容は十分な分量を確保し、箇条書き3行程度で終わらせず、深い分析と具体的な提案を含めること。

# 図表・グラフの活用ルール

contentフィールドのマークダウン内に、以下の2種類のコードブロックを使って図表やグラフを含めてください。
テキストだけでなく、視覚的な図表を積極的に活用し、説得力のある資料にすること。

## Mermaid図（フローチャート、関係図など）
```mermaid
graph TD
    A["ステップ1"] --> B["ステップ2"]
    B --> C["ステップ3"]
```

## Chart.jsグラフ（棒グラフ、円グラフ、折れ線グラフなど）
```chart
{
  "type": "bar",
  "data": {
    "labels": ["ラベル1", "ラベル2"],
    "datasets": [{
      "label": "データセット名",
      "data": [10, 20],
      "backgroundColor": ["#2563eb", "#ef4444"]
    }]
  },
  "options": {
    "responsive": true,
    "maintainAspectRatio": true
  }
}
```

## スライドタイプ別の推奨図表
- **analysis**: 市場シェアの円グラフ（chart/pie）、競合比較の棒グラフ（chart/bar）
- **proposal**: ターゲットセグメントの関係図（mermaid/graph LR）、機会の大きさのドーナツ（chart/doughnut）
- **flow**: 営業プロセスのフローチャート（mermaid/graph TD）、タイムライン（chart/line）
- **pricing**: 料金プラン比較の棒グラフ（chart/bar）、ROI推移（chart/line）
- **summary**: KPI改善予測の棒グラフ（chart/bar）

## 図表の注意事項
- 1つのスライドに図表は最大1つまで（テキスト説明と組み合わせる）
- Mermaidのノードテキストに括弧や特殊記号がある場合は["引用符"]で囲むこと
- Chart.jsのJSONは厳密なJSON形式（末尾カンマ禁止、キーはダブルクォート）で書くこと
- 色は #2563eb（青）、#ef4444（赤）、#22c55e（緑）、#f59e0b（黄）、#8b5cf6（紫）、#0ea5e9（水色）を使うこと
- すべてのスライドに図表が必要なわけではない。テキストの方が適切な場合はテキストのみでよい
- 図表を使う場合は、図表の前後にテキストでの説明も必ず含めること

JSON配列のみを出力し、それ以外のテキストは含めないこと。
出力は ```json で囲まず、JSON配列をそのまま直接出力すること。[ で始まり ] で終わること。"""


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


def _extract_product_info(input_type, form, files):
    """リクエストから商品情報テキストを抽出する共通処理"""
    if input_type == "url":
        url = form.get("url", "").strip()
        if not url:
            raise ValueError("URLを入力してください")
        return extract_text_from_url(url)
    elif input_type == "pdf":
        if "pdf_file" not in files:
            raise ValueError("PDFファイルを選択してください")
        pdf_file = files["pdf_file"]
        if pdf_file.filename == "":
            raise ValueError("PDFファイルを選択してください")
        return extract_text_from_pdf(pdf_file)
    elif input_type == "text":
        text = form.get("text_input", "").strip()
        if not text:
            raise ValueError("商品情報を入力してください")
        return text
    else:
        raise ValueError("無効な入力タイプです")


def analyze_product(api_key, product_info):
    """商品情報をAIに分析させ、認識結果をJSONで返す"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": ANALYZE_PROMPT},
            {
                "role": "user",
                "content": f"以下の商品・サービス情報を分析してください。\n\n{product_info}",
            },
        ],
        temperature=0.5,
        max_tokens=2000,
    )
    content = response.choices[0].message.content.strip()
    content = _extract_json_object(content)
    return json.loads(content)


def _extract_json_object(text):
    """テキストからJSONオブジェクト部分を安全に抽出する。"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def generate_slides(api_key, product_info, analysis_context=None, custom_prompt=None):
    """OpenAI APIで提案資料スライドを生成"""
    role_prompt = custom_prompt if custom_prompt else DEFAULT_ROLE_PROMPT
    system_prompt = role_prompt + "\n" + SYSTEM_OUTPUT_RULES

    user_message = "以下の商品・サービス情報をもとに、営業提案資料のスライドを作成してください。\n\n"
    if analysis_context:
        user_message += f"【AIによる事前分析（確認済み）】\n{analysis_context}\n\n"
    user_message += f"【元の商品情報】\n{product_info}"

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=8192,
    )
    content = response.choices[0].message.content.strip()
    # AIの出力からJSON配列を抽出
    # content内に```mermaid```や```chart```が含まれるため、
    # 単純なsplit("```")では壊れる。JSONの [ ] で範囲を特定する。
    content = _extract_json_array(content)
    return json.loads(content)


def _extract_json_array(text):
    """テキストからJSON配列部分を安全に抽出する。
    コードブロック内にバッククォートが含まれていても正しく動作する。"""
    # まず ```json で始まるコードブロックを除去（最外殻のみ）
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
        # 末尾の ``` を除去
        if text.endswith("```"):
            text = text[:-3].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # JSON配列の先頭 [ と末尾 ] を探す
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/default-prompt")
def default_prompt():
    """デフォルトプロンプトを返すAPI"""
    return jsonify({"prompt": DEFAULT_ROLE_PROMPT})


@app.route("/analyze", methods=["POST"])
def analyze():
    """Step 1: 商品認識・市場確認"""
    input_type = request.form.get("input_type", "text")
    api_key = request.form.get("api_key", "").strip()

    if not api_key:
        return jsonify({"error": "APIキーが設定されていません。画面右上の歯車アイコンから設定してください。"}), 400

    try:
        product_info = _extract_product_info(input_type, request.form, request.files)

        if len(product_info.strip()) < 10:
            return jsonify({"error": "商品情報が短すぎます。もう少し詳しい情報を入力してください"}), 400

        analysis = analyze_product(api_key, product_info)
        return jsonify({"analysis": analysis, "raw_text": product_info})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except requests.RequestException as e:
        return jsonify({"error": f"URLの取得に失敗しました: {str(e)}"}), 400
    except json.JSONDecodeError:
        return jsonify({"error": "AIからの応答の解析に失敗しました。もう一度お試しください"}), 500
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return jsonify({"error": "APIキーが無効です。正しいキーを設定してください。"}), 401
        return jsonify({"error": f"エラーが発生しました: {error_msg}"}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """Step 2: 提案資料スライド生成"""
    api_key = request.json.get("api_key", "").strip() if request.is_json else request.form.get("api_key", "").strip()

    if not api_key:
        return jsonify({"error": "APIキーが設定されていません。"}), 400

    try:
        if request.is_json:
            data = request.json
            product_info = data.get("raw_text", "")
            analysis_context = data.get("analysis_context", "")
            custom_prompt = data.get("custom_prompt", "").strip() or None
        else:
            product_info = request.form.get("raw_text", "")
            analysis_context = request.form.get("analysis_context", "")
            custom_prompt = request.form.get("custom_prompt", "").strip() or None

        if not product_info:
            return jsonify({"error": "商品情報がありません"}), 400

        slides = generate_slides(api_key, product_info, analysis_context, custom_prompt)
        return jsonify({"slides": slides})

    except json.JSONDecodeError:
        return jsonify({"error": "AIからの応答の解析に失敗しました。もう一度お試しください"}), 500
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            return jsonify({"error": "APIキーが無効です。正しいキーを設定してください。"}), 401
        return jsonify({"error": f"エラーが発生しました: {error_msg}"}), 500


@app.route("/slides")
def slides():
    return render_template("slides.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
