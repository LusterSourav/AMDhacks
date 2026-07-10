"""Local testing server: POST /api/chat with {"query": "..."}"""

from flask import Flask, request, jsonify
from main import route

app = Flask(__name__)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("query", "")
    result = route(query)
    return jsonify(result)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
