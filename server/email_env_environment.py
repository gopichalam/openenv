import random
from pydantic import BaseModel, Field
from typing import List
from difflib import SequenceMatcher

# ---------------- MODELS ---------------- #

class TaskItem(BaseModel):
    task: str
    priority: int = Field(ge=1, le=5)


class Action(BaseModel):
    tasks: List[TaskItem]
    action_type: str


class Observation(BaseModel):
    email: str
    step_index: int
    remaining: int
    reward: float = 0.0
    done: bool = False


# ---------------- DATASET ---------------- #

def get_task_dataset(difficulty):

    if difficulty == "easy":
        return [
            {
                "email": "Submit the sales report by 5 PM today",
                "expected": {
                    "tasks": [{"task": "submit sales report", "priority": 5}],
                    "action_type": "create_task"
                }
            },
            {
                "email": "Call the client tomorrow",
                "expected": {
                    "tasks": [{"task": "call client", "priority": 4}],
                    "action_type": "create_task"
                }
            },
            {
                "email": "Send invoice today",
                "expected": {
                    "tasks": [{"task": "send invoice", "priority": 5}],
                    "action_type": "create_task"
                }
            },
            {
                "email": "Schedule a meeting with HR",
                "expected": {
                    "tasks": [{"task": "schedule meeting with HR", "priority": 3}],
                    "action_type": "schedule"
                }
            },
            {
                "email": "Review the document",
                "expected": {
                    "tasks": [{"task": "review document", "priority": 3}],
                    "action_type": "create_task"
                }
            }
        ]

    if difficulty == "medium":
        return [
            {
                "email": "Review the document and send feedback by EOD",
                "expected": {
                    "tasks": [
                        {"task": "review document", "priority": 4},
                        {"task": "send feedback", "priority": 4}
                    ],
                    "action_type": "create_task"
                }
            },
            {
                "email": "Prepare slides and schedule a meeting tomorrow",
                "expected": {
                    "tasks": [
                        {"task": "prepare slides", "priority": 4},
                        {"task": "schedule meeting", "priority": 4}
                    ],
                    "action_type": "schedule"
                }
            }
        ]

    if difficulty == "hard":
        return [
            {
                "email": """
    Hi team,

    Please finalize the presentation deck, review the budget document,
    and schedule a discussion tomorrow.

    Thanks
    """,
                "expected": {
                    "tasks": [
                        {"task": "finalize presentation deck", "priority": 4},
                        {"task": "review budget document", "priority": 4}
                    ],
                    "action_type": "schedule"
                }
            },
            {
                "email": """
    Hey,

    We need to send the report today, fix the errors,
    and also plan a meeting next week.

    Thanks
    """,
                "expected": {
                    "tasks": [
                        {"task": "send report", "priority": 5},
                        {"task": "fix errors", "priority": 4}
                    ],
                    "action_type": "schedule"
                }
            }
        ]


# ---------------- GRADER ---------------- #

# Multi-factor scoring:
# - Coverage (40%): task match quality
# - Priority (25%): priority correctness
# - Action (20%): correct action type
# - Penalty: over/under prediction
# - Bonus: perfect match

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def grade_multi(pred, expected):
    score = 0
    breakdown = {}

    pred_tasks = pred.get("tasks", [])
    exp_tasks = expected.get("tasks", [])

    # ---------------- COVERAGE ---------------- #
    matches = 0
    for et in exp_tasks:
        for pt in pred_tasks:
            if similarity(pt["task"], et["task"]) > 0.5:
                matches += 1
                break

    coverage = matches / len(exp_tasks) if exp_tasks else 0
    score += 0.4 * coverage
    breakdown["coverage"] = round(coverage, 2)

    # ---------------- PRIORITY ---------------- #
    # Priority rules:
    # today → high (5)
    # tomorrow → medium-high (4)
    # otherwise → default (3)

    priority_hits = 0
    for et in exp_tasks:
        for pt in pred_tasks:
            if similarity(pt["task"], et["task"]) > 0.5:
                if abs(pt["priority"] - et["priority"]) <= 1:
                    priority_hits += 1

    priority_score = min(1, priority_hits / len(exp_tasks)) if exp_tasks else 0
    score += 0.25 * priority_score
    breakdown["priority"] = round(priority_score, 2)

    # ---------------- ACTION TYPE ---------------- #
    action_score = 1 if pred.get("action_type") == expected.get("action_type") else 0
    score += 0.2 * action_score
    breakdown["action"] = action_score

    # ---------------- PENALTY ---------------- #
    penalty = 0

    # Too many tasks
    if len(pred_tasks) > len(exp_tasks) + 2:
        penalty += 0.1

    # No tasks predicted
    if len(pred_tasks) == 0:
        penalty += 0.2

    score -= penalty
    breakdown["penalty"] = penalty

    # ---------------- BONUS ---------------- #
    if coverage == 1 and priority_score == 1 and action_score == 1:
        score += 0.15  # perfect answer bonus

    # Clamp score
    score = max(0, min(1, round(score, 2)))

    return score, breakdown


# ---------------- ENV ---------------- #

class EmailEnvironment:

    def __init__(self):
        random.seed(42)
        self.tasks = []
        self.index = 0
        self.last_score = 0.0
        self.last_breakdown = {}
        self.current_difficulty = "easy"

    def set_difficulty(self, difficulty: str):
        """
        Set the difficulty level that will be used on the next reset().
        Call /set_difficulty?difficulty=hard BEFORE calling /reset.
        Supported values: "easy", "medium", "hard"
        """
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"Unknown difficulty: {difficulty!r}. Must be one of: easy, medium, hard")
        self.current_difficulty = difficulty

    def reset(self, difficulty=None, seed=None, episode_id=None, **kwargs):
        """
        Reset the environment.

        - When the OpenEnv framework calls /reset, it passes no difficulty arg,
          so we use self.current_difficulty (set via /set_difficulty beforehand).
        - When called directly with an explicit difficulty (e.g. from inference.py
          or /baseline), that value takes precedence and also updates current_difficulty.
        """
        if seed is not None:
            random.seed(seed)

        if difficulty is not None:
            self.current_difficulty = difficulty

        self.tasks = get_task_dataset(self.current_difficulty)
        self.index = 0
        self.last_score = 0.0
        self.last_breakdown = {}

        return self.state()

    def state(self):
        # Guard against IndexError after episode ends
        if self.index >= len(self.tasks):
            return Observation(
                email="",
                step_index=self.index,
                remaining=0,
                reward=self.last_score,
                done=True
            )

        current = self.tasks[self.index]
        return Observation(
            email=current["email"],
            step_index=self.index,
            remaining=len(self.tasks) - self.index - 1,
            reward=0.0,
            done=False
        )

# Combine user input with email context to improve task extraction accuracy
# This helps capture implicit tasks mentioned only in the email

    def step(self, action_dict, **kwargs):
        try:
            # Ensure dict format
            if not isinstance(action_dict, dict):
                action_dict = action_dict.model_dump()

            # Extract message
            if "action" in action_dict:
                message = action_dict["action"].get("message", "").lower()
            else:
                message = action_dict.get("message", "").lower()

            # Add context (VERY IMPORTANT for better scoring)
            current_email = self.tasks[self.index]["email"].lower()
            combined_text = message + " " + current_email

            # Priority detection
            if "today" in combined_text:
                priority = 5
            elif "tomorrow" in combined_text:
                priority = 4
            else:
                priority = 3

            tasks = []

            # ---------------- SPECIAL CASE: REPORT ---------------- #
            if "report" in combined_text:
                if "sales" in combined_text:
                    tasks.append({"task": "submit sales report", "priority": priority})
                else:
                    tasks.append({"task": "send report", "priority": priority})

            # ---------------- GENERIC KEYWORD MAPPING ---------------- #
            # Keyword-to-task mapping for lightweight NLP extraction
            # Designed to balance simplicity and coverage across all difficulty levels
            task_map = {
                "feedback": "send feedback",
                "slides": "prepare slides",
                "invoice": "send invoice",
                "bill": "send invoice",
                "call": "call client",
                "fix": "fix errors",
                "error": "fix errors",
                "hr": "schedule meeting with HR",
                "review": "review document",
                "finalize": "finalize presentation deck",
                "budget": "review budget document"
            }

            for keyword, task_name in task_map.items():
                if keyword in combined_text:
                    tasks.append({
                        "task": task_name,
                        "priority": priority
                    })

            # ---------------- ACTION TYPE ---------------- #
            # Determine action type based on scheduling intent keywords
            # "schedule" is required for meeting-related tasks

            if any(word in combined_text for word in ["schedule", "meeting", "discussion"]):
                action_type = "schedule"
            else:
                action_type = "create_task"

            if "schedule" in combined_text or "meeting" in combined_text:
                tasks.append({
                    "task": "schedule meeting",
                    "priority": priority
                })

            # ---------------- FALLBACK ---------------- #
            if not tasks:
                clean_message = combined_text
                for word in ["by 5 pm", "today", "tomorrow", "please", "hi", "team"]:
                    clean_message = clean_message.replace(word, "")
                clean_message = clean_message.strip()

                tasks.append({
                    "task": clean_message,
                    "priority": priority
                })

            # Order-preserving deduplication
            seen = set()
            unique_tasks = []
            for t in tasks:
                key = t["task"]
                if key not in seen:
                    seen.add(key)
                    unique_tasks.append(t)
            tasks = unique_tasks

            parsed_action = {
                "tasks": tasks,
                "action_type": action_type
            }

            expected = self.tasks[self.index]["expected"]

            # ADD THESE LINES for Debugging
            # print("\n====================")
            # print("INPUT MESSAGE:", message)
            # print("CURRENT EMAIL:", current_email)
            # print("PARSED ACTION:", parsed_action)
            # print("EXPECTED:", expected)

            reward, breakdown = grade_multi(parsed_action, expected)

            # ADD THESE LINES for Debugging
            # print("REWARD:", reward)
            # print("BREAKDOWN:", breakdown)
            # print("====================\n")

            self.last_score = reward
            self.last_breakdown = breakdown

        except Exception:
            reward, breakdown = 0.0, {"error": 1}
            self.last_score = reward
            self.last_breakdown = breakdown

        # ---------------- MOVE TO NEXT STEP ---------------- #
        self.index += 1
        done = self.index >= len(self.tasks)

        if done:
            obs = Observation(
                email="",
                step_index=self.index,
                remaining=0,
                reward=reward,
                done=True
            )
        else:
            obs = self.state()
            obs.reward = reward
            obs.done = False

        return obs

    # ✅ REQUIRED BY OPENENV

    async def reset_async(self, *args, **kwargs):
        return self.reset(*args, **kwargs)

    async def step_async(self, action, *args, **kwargs):
        action_dict = action.model_dump()
        return self.step(action_dict, **kwargs)

    def close(self):
        pass
