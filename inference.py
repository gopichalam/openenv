import os
import json
import requests
from openai import OpenAI

# ================================
# REQUIRED ENV VARIABLES
# ================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
HF_TOKEN     = os.getenv("HF_TOKEN")

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# ================================
# INIT CLIENT (MANDATORY)
# ================================

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ================================
# HELPERS
# ================================

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


def env_post(path: str, payload: dict = None):
    url = f"{ENV_URL}{path}"
    r = requests.post(url, json=payload or {})
    r.raise_for_status()
    return r.json()


def env_get(path: str):
    url = f"{ENV_URL}{path}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


# ================================
# RUN ONE DIFFICULTY EPISODE
# ================================

def run_episode(difficulty: str) -> float:
    # Set difficulty then reset via standard /reset endpoint
    requests.post(f"{ENV_URL}/set_difficulty", params={"difficulty": difficulty})
    obs_data = env_post("/reset")

    episode_rewards = []
    step_num = 0

    while True:
        # Build observation from current state
        if isinstance(obs_data, dict):
            email    = obs_data.get("email", "")
            done     = obs_data.get("done", False)
            reward   = obs_data.get("reward", 0.0)
            remaining = obs_data.get("remaining", 0)
        else:
            break

        if done:
            episode_rewards.append(reward)
            break

        # Build prompt for LLM
        prompt = f"""You are an email task extractor. Given this email, extract the tasks as a short natural-language list.

Email:
{email}

Respond with a single sentence listing the tasks and whether they need scheduling. Example:
"review document and send feedback, action: create_task"
"""

        llm_response = call_llm(prompt)

        # Mandatory [START] log
        print(json.dumps({
            "type": "[START]",
            "difficulty": difficulty,
            "step": step_num,
            "email": email,
            "remaining": remaining
        }))

        # Submit action
        action = {"message": llm_response}
        obs_data = env_post("/step", {"action": action})

        step_reward = obs_data.get("reward", 0.0) if isinstance(obs_data, dict) else 0.0
        episode_rewards.append(step_reward)

        # Mandatory [STEP] log
        print(json.dumps({
            "type": "[STEP]",
            "difficulty": difficulty,
            "step": step_num,
            "action": llm_response,
            "reward": step_reward,
            "done": obs_data.get("done", False) if isinstance(obs_data, dict) else True
        }))

        step_num += 1

        if isinstance(obs_data, dict) and obs_data.get("done", False):
            break

    episode_score = round(sum(episode_rewards) / len(episode_rewards), 4) if episode_rewards else 0.0

    # Mandatory [END] log
    print(json.dumps({
        "type": "[END]",
        "difficulty": difficulty,
        "episode_score": episode_score,
        "steps": step_num
    }))

    return episode_score


# ================================
# ENTRY POINT
# ================================

if __name__ == "__main__":
    difficulties = ["easy", "medium", "hard"]
    all_scores = []

    for diff in difficulties:
        score = run_episode(diff)
        all_scores.append(score)
        print(json.dumps({
            "type": "[RESULT]",
            "difficulty": diff,
            "score": score
        }))

    overall = round(sum(all_scores) / len(all_scores), 4)
    print(json.dumps({
        "type": "[FINAL]",
        "overall_score": overall,
        "scores": dict(zip(difficulties, all_scores))
    }))
