from openai import OpenAI


BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
MODEL = "qwen2.5:14b-instruct"


def main() -> int:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    # Enable streaming
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Say hello in one short sentence."}
        ],
        temperature=0.2,
        max_tokens=50,
        stream=True  # 1. Turn on streaming
    )

    print("\n--- Qwen3 Response ---")
    # 2. Iterate through the chunks as they arrive from the cluster
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            
    print("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
