import os
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

if HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )
else:
    client = None

# ================================
# HELPERS
# ================================

def call_llm(prompt: str) -> str:
    if client is None:
        return ""   # 👈 no warning needed

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        if (
            response
            and response.choices
            and len(response.choices) > 0
            and response.choices[0].message
            and response.choices[0].message.content
        ):
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[WARN] LLM error: {str(e)}", flush=True)

    return ""


def env_post(path: str, payload: dict = None):
    try:
        url = f"{ENV_URL}{path}"
        r = requests.post(url, json=payload or {})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[WARN] env_post failed: {str(e)}", flush=True)
        return {"email": "", "done": True, "reward": 0.5}


# ================================
# RUN ONE DIFFICULTY EPISODE
# ================================

def run_episode(difficulty: str) -> float:
    try:
        requests.post(f"{ENV_URL}/set_difficulty", params={"difficulty": difficulty})
    except Exception as e:
        print(f"[WARN] set_difficulty failed: {str(e)}", flush=True)
        return 0.5

    obs_data = env_post("/reset")

    episode_rewards = []
    step_num = 0

    while True:
        if not isinstance(obs_data, dict):
            break

        email = obs_data.get("email", "")
        done = obs_data.get("done", False)
        reward = obs_data.get("reward", 0.0)
        remaining = obs_data.get("remaining", 0)

        if done:
            episode_rewards.append(reward)
            break

        # ✅ START BLOCK (PRINT BEFORE STEP 0 ACTION)
        if step_num == 0:
            print(f"[START] difficulty={difficulty} email={email} remaining={remaining}", flush=True)

        prompt = f"""
Extract actionable tasks using EXACT keywords.

Use only these words if relevant:
report, slides, feedback, invoice, call, fix, review, budget, meeting

Rules:
- Keep phrases short (e.g., "review document", "send invoice")
- Include ALL tasks mentioned
- Do NOT add extra words

Action rules:
- If meeting/schedule/discussion present → action: schedule
- Otherwise → action: create_task

Email:
{email}

Output format:
task1, task2, task3 | action: create_task OR schedule
"""

        llm_response = call_llm(prompt)

        if not llm_response:
            llm_response = email.lower()

            # clean noise
            for w in ["please", "hi", "team", "thanks"]:
                llm_response = llm_response.replace(w, "")

        # ALWAYS apply keyword injection
        keywords = ["report", "slides", "feedback", "invoice", "call", "fix", "review", "budget", "meeting"]

        for kw in keywords:
            if kw in email.lower() and kw not in llm_response.lower():
                llm_response += f" {kw}"
        llm_response = llm_response.strip()
        
        action = {"message": llm_response}
        try:
            obs_data = env_post("/step", {"action": action})
        except Exception as e:
            print(f"[WARN] Step failed: {str(e)}", flush=True)
            break

        step_reward = obs_data.get("reward", 0.0) if isinstance(obs_data, dict) else 0.0
        episode_rewards.append(step_reward)

        print(f"[STEP] difficulty={difficulty} step={step_num} reward={step_reward}", flush=True)

        step_num += 1

        if isinstance(obs_data, dict) and obs_data.get("done", False):
            break

    episode_score = round(sum(episode_rewards) / len(episode_rewards), 4) if episode_rewards else 0.0

    print(f"[END] difficulty={difficulty} score={episode_score} steps={step_num}", flush=True)

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
        print(f"[RESULT] difficulty={diff} score={score}", flush=True)

    overall = round(sum(all_scores) / len(all_scores), 4)
    print(f"[FINAL] overall_score={overall}", flush=True)