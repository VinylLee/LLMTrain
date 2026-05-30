#!/usr/bin/env python3
"""简单脚本：测试LM Studio本地API连接性"""

import os
import sys
import json
import requests

# 手动加载.env文件（不需要dotenv依赖）
def load_env_file(path='.env'):
    """手动加载.env文件"""
    if not os.path.exists(path):
        return
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip()

load_env_file()


def test_deepseek_connection():
    # 使用前面加载的环境变量

    # 对于LM Studio，使用LMSTUDIO配置；对于DeepSeek，使用DEEPSEEK配置
    api_key = os.getenv("LMSTUDIO_API_KEY") or os.getenv("DEEPSEEK_KEY")
    api_url = os.getenv("LMSTUDIO_BASE_URL") or os.getenv("DEEPSEEK_OPENAI_BASE_URL")
    model = os.getenv("LMSTUDIO_QWEN") or os.getenv("DEEPSEEK_V4_FLASH") or "qwen3.6"

    print("=" * 60)
    print("API Connection Test")
    print("=" * 60)
    print(f"API Base URL: {api_url}")
    print(f"Model:        {model}")
    print(f"Has API Key:  {bool(api_key)}\n")

    # 验证必要参数
    if not api_key:
        print("\n⚠️  Note: No API key found (OK for LM Studio)")
        api_key = ""

    if not api_url:
        print("\n❌ Error: API Base URL not found in .env")
        return False

    # 构建完整的API URL
    full_url = api_url.rstrip("/")
    if not full_url.endswith("/v1/chat/completions"):
        if full_url.endswith("/v1"):
            full_url = full_url + "/chat/completions"
        else:
            full_url = full_url + "/v1/chat/completions"

    print(f"Full URL:     {full_url}\n")

    # 构建请求头和payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello, are you working? Please reply with just 'Yes'."}
        ],
        "max_tokens": 50,
    }

    print("Sending request to DeepSeek API...")
    try:
        response = requests.post(full_url, json=payload, headers=headers, timeout=30)
        status_code = response.status_code

        print(f"Status Code:  {status_code}")

        if status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS: API is reachable!")
            print("\nResponse:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # 提取模型回复
            if "choices" in data and len(data["choices"]) > 0:
                message = data["choices"][0].get("message", {})
                content = message.get("content", "")
                print(f"\nModel response: {content}")

            # 打印token用量
            if "usage" in data:
                usage = data["usage"]
                print(f"\nToken usage:")
                print(f"  Prompt tokens:     {usage.get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  Total tokens:      {usage.get('total_tokens', 'N/A')}")

            return True
        else:
            print(f"\n❌ Error: HTTP {status_code}")
            try:
                error_data = response.json()
                print("Error response:")
                print(json.dumps(error_data, indent=2, ensure_ascii=False))
            except:
                print("Error response:")
                print(response.text)
            return False

    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timeout (30s)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Error: Connection failed - {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = test_deepseek_connection()
    sys.exit(0 if success else 1)
