# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Email Env Environment.

Difficulty flow:
    1. POST /set_difficulty?difficulty=hard   ← set difficulty FIRST
    2. POST /reset                            ← framework reset picks it up
    3. POST /step  (repeat)
    4. GET  /grader                           ← check last score

Endpoints:
    - POST /reset            Reset the environment (OpenEnv standard)
    - POST /step             Execute an action
    - GET  /state            Get current environment state
    - GET  /schema           Get action/observation schemas
    - POST /set_difficulty   Set difficulty before reset (easy/medium/hard)
    - GET  /baseline         Run a quick baseline across all difficulties
    - GET  /tasks            List available difficulty levels
    - GET  /grader           Get last score and breakdown
    - WS   /ws               WebSocket endpoint for persistent sessions
"""
# Handle both execution modes:
# 1. Package execution (e.g., uv run server)
# 2. Direct module execution (e.g., uvicorn server.app:app)
# This ensures compatibility across local dev and evaluation environments

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # Package-style import (uv run, production)
    from ..models import EmailAction, EmailObservation
    from .email_env_environment import EmailEnvironment
except ImportError:
    # Direct execution (uvicorn, local dev)
    from models import EmailAction, EmailObservation
    from server.email_env_environment import EmailEnvironment

env_instance = EmailEnvironment()

app = create_app(
    lambda: env_instance,
    EmailAction,
    EmailObservation,
    env_name="email_env",
    max_concurrent_envs=1,
)

from fastapi import Request, HTTPException

# Difficulty must be set BEFORE reset()
# Reset uses current_difficulty to load appropriate dataset

#     """
#     Set the difficulty level that will be used on the next /reset call.
#     Must be called BEFORE /reset.

#     Query param:
#         difficulty: "easy" | "medium" | "hard"  (default: "easy")

#     Example:
#         POST /set_difficulty?difficulty=hard
#         POST /reset
#     """


from fastapi import Query, HTTPException

@app.post("/set_difficulty")
async def set_difficulty(difficulty: str = Query("easy")):
    if difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid difficulty '{difficulty}'. Must be one of: easy, medium, hard"
        )
    env_instance.set_difficulty(difficulty)
    return {
        "status": "ok",
        "difficulty": difficulty,
        "message": f"Difficulty set to '{difficulty}'. Now call /reset to start the episode."
    }
# Baseline uses a single generic message covering all task types
# Helps evaluate environment performance across difficulty levels

@app.get("/baseline")
def run_baseline():
    """Run a single-step baseline across all three difficulty levels."""
    scores = []

    for difficulty in ["easy", "medium", "hard"]:
        env_instance.reset(difficulty=difficulty)

        action = {
            "message": (
                "submit sales report today call client tomorrow send invoice "
                "review document prepare slides send feedback "
                "finalize presentation review budget document "
                "schedule meeting"
            )
}

        obs = env_instance.step(action)
        scores.append(obs.reward)

    return {
        "baseline_score": round(sum(scores) / len(scores), 2),
        "details": {"easy": scores[0], "medium": scores[1], "hard": scores[2]}
    }


@app.get("/tasks")
def get_tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "current_difficulty": env_instance.current_difficulty,
        "action_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Natural language email input to extract tasks from"
                }
            },
            "required": ["message"]
        }
    }


@app.get("/grader")
def get_grader():
    return {
        "score": env_instance.last_score,
        "breakdown": env_instance.last_breakdown
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
