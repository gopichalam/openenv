import os
from openai import OpenAI

# ================================
# REQUIRED ENV VARIABLES
# ================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional (for local docker)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ================================
# INIT CLIENT (MANDATORY)
# ================================

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ================================
# SIMPLE INFERENCE FUNCTION
# ================================

def run_inference(prompt: str):
    print("START inference")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    output = response.choices[0].message.content

    print("STEP model_response:", output)

    print("END inference")

    return output


# ================================
# ENTRY POINT
# ================================

if __name__ == "__main__":
    test_prompt = "Extract tasks from: Submit report today"
    result = run_inference(test_prompt)
    print(result)