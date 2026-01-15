from flask import Flask, request, Response, stream_with_context
import requests
import json
from datetime import datetime

app = Flask(__name__)
VLLM_URL = "http://localhost:8000"


# =========================
# Utils
# =========================

def has_triple_hash(data):
    """Kiểm tra xem request có chứa ### không"""
    if not data:
        return False

    if isinstance(data, dict) and 'messages' in data:
        for msg in data.get('messages', []):
            content = msg.get('content', '')
            if '###' in content:
                return True

    return False


def log_json_safely(data, label=""):
    """Log JSON data một cách an toàn"""
    try:
        print(label)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print(f"{label}: {str(data)[:500]}")


def remove_function_tools(data: dict) -> dict:
    """
    Remove all tools with type == 'function'
    """
    if not isinstance(data, dict):
        return data

    tools = data.get("tools")
    if not isinstance(tools, list):
        return data
    print("List tools:", tools)
    filtered_tools = [
        t for t in tools
        if not (isinstance(t, dict) and t.get("type") == "function")
    ]

    if filtered_tools:
        data["tools"] = filtered_tools
    else:
        data.pop("tools", None)

    return data


# =========================
# Proxy
# =========================

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f"{VLLM_URL}/{path}"

    request_data = None
    body_data = None
    is_streaming = False

    # -------------------------
    # Parse & sanitize body
    # -------------------------
    if request.method == 'POST' and request.is_json:
        request_data = request.get_json()
        is_streaming = request_data.get('stream', False)

        # Remove function tools
        request_data = remove_function_tools(request_data)

        body_data = json.dumps(request_data).encode("utf-8")
    else:
        body_data = request.get_data()

    # -------------------------
    # Logging
    # -------------------------
    skip_log = has_triple_hash(request_data)

    if skip_log:
        print(f"[{datetime.now()}] SKIPPED: Request with ### in content")
    else:
        print(f"\n{'=' * 100}")
        print(f"[{datetime.now()}] {request.method} {url}")

        if request_data:
            log_json_safely(request_data, "INPUT:")

            if "tools" in request_data:
                print(f"TOOLS AFTER FILTER: {len(request_data['tools'])}")
            else:
                print("TOOLS AFTER FILTER: none")

    # -------------------------
    # Forward request
    # -------------------------
    try:
        headers = {
            k: v for k, v in request.headers
            if k.lower() not in ['host', 'content-length']
        }

        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=body_data,
            stream=True,
            timeout=300
        )

        # -------------------------
        # Streaming response
        # -------------------------
        if is_streaming:
            if not skip_log:
                print(f"\nSTREAMING RESPONSE ({resp.status_code}):")
                print("(Streaming data - showing first chunks)")

            def generate():
                chunk_count = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        if not skip_log and chunk_count < 5:
                            try:
                                print(chunk.decode("utf-8")[:200])
                            except Exception:
                                pass
                            chunk_count += 1
                        yield chunk

                if not skip_log:
                    print("=" * 100 + "\n")

            return Response(
                stream_with_context(generate()),
                status=resp.status_code,
                headers=dict(resp.headers),
            )

        # -------------------------
        # Non-stream response
        # -------------------------
        content = resp.content

        if not skip_log:
            print(f"\nOUTPUT ({resp.status_code}):")
            if 'application/json' in resp.headers.get('content-type', ''):
                try:
                    log_json_safely(resp.json(), "")
                except Exception:
                    print(content.decode("utf-8")[:1000])
            else:
                print(content.decode("utf-8")[:500])
            print("=" * 100 + "\n")

        return Response(content, resp.status_code, dict(resp.headers))

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}\n")
        return Response(str(e), 502)


# =========================
# Entrypoint
# =========================

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8001,
        debug=False,
        threaded=True
    )
