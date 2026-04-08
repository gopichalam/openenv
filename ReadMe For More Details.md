# Email Task Extraction Environment (OpenEnv Compatible)

## 📌 Overview

This project implements an email task extraction environment using OpenEnv and FastAPI.
It processes natural language email content and extracts structured tasks with priorities and action types.

The system supports multiple difficulty levels and includes a grading mechanism to evaluate performance.

---

## 🚀 Features

* Multi-difficulty support: Easy, Medium, Hard
* Natural language task extraction
* Priority detection (today, tomorrow, default)
* Action type classification (create_task / schedule)
* Built-in grading system with detailed breakdown
* OpenEnv-compatible API endpoints

---

## 🧠 Approach

### Task Extraction Logic

* Combines user input + email context for better accuracy
* Uses keyword-based mapping for:

  * report, slides, feedback, invoice, call, fix, HR, review, finalize, budget
* Handles multi-task extraction from a single message

---

### Priority Detection

* "today" → priority 5
* "tomorrow" → priority 4
* default → priority 3

---

### Action Type Detection

* "schedule", "meeting", "discussion" → schedule
* otherwise → create_task

---

### Grading Strategy

Score components:

* Coverage (40%) → task match
* Priority (25%) → correctness
* Action Type (20%) → correctness
* Penalty → extra/missing tasks
* Bonus → perfect match

---

## 🛠️ Setup Instructions

### 1. Install dependencies

```bash
pip install fastapi uvicorn pydantic
```

---

### 2. Run the server

```bash
uvicorn server.app:app
```

---

### 3. Open API docs

```
http://127.0.0.1:8000/docs
```

---

## 🔄 API Usage Flow

### Step 1: Set difficulty

```bash
POST /set_difficulty?difficulty=easy
```

### Step 2: Reset environment

```bash
POST /reset
```

### Step 3: Send action

```json
{
  "action": {
    "message": "submit sales report today"
  }
}
```

### Step 4: Get score

```bash
GET /grader
```

---

## 🧪 Example Test Cases

### Easy

* "submit sales report today"
* "call client tomorrow"

### Medium

* "review document and send feedback"
* "prepare slides and schedule meeting tomorrow"

### Hard

* "finalize presentation and review budget"
* "send report today and fix errors"

---

## 📊 Baseline Evaluation

Run:

```bash
GET /baseline
```

Provides average performance across all difficulty levels.

---

## 📁 Project Structure

```
email_env/
│
├── models.py
├── server/
│   ├── app.py
│   ├── email_env_environment.py
├── ReadMe For More Details.md
```

---

## ⚠️ Notes

* Import handling supports both:

  * package execution (`uv run`)
  * direct execution (`uvicorn`)
* Designed for compatibility with OpenEnv evaluation

---

## ✅ Status

* All difficulty levels tested and validated
* Scoring system verified
* Submission-ready

---
