import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openrouter/auto"
DATA_FILE = "attendance.json"


# ---------- Database Functions ----------

def load_data():
    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8-sig") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print("Database error:", error)
        return []


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def normalize_roll(roll):
    return str(roll).strip().lstrip("0") or "0"


def find_student(students, roll):
    if roll is None:
        return None

    for student in students:
        if normalize_roll(student.get("roll", "")) == normalize_roll(roll):
            return student

    return None


# ---------- Attendance Calculation ----------

def calculate_attendance(student):
    attendance = student.get("attendance_5_days", [])

    recorded = [
        day for day in attendance
        if day in ("Present", "Absent")
    ]

    present = recorded.count("Present")
    total = len(recorded)
    percentage = present / total * 100 if total else 0

    return present, total, percentage


# ---------- Execute AI Commands ----------

def execute_action(command):
    if not isinstance(command, dict):
        return {"error": "Invalid AI command."}

    action = command.get("action")

    if action == "clarify":
        return {
            "ai_question": command.get(
                "question", "Please provide more details."
            )
        }

    students = load_data()

    if action == "get_all_attendance":
        return {"students": students}

    if action == "get_student":
        roll = command.get("roll") or command.get("roll_no")
        student = find_student(students, roll)

        if student is None:
            return {"error": f"Student with roll number {roll} not found."}

        present, total, percentage = calculate_attendance(student)

        return {
            "roll": student["roll"],
            "name": student["name"],
            "present_days": present,
            "recorded_days": total,
            "attendance_percentage": f"{percentage:.1f}%",
            "status": "Eligible" if percentage >= 75
                      else "Short Attendance Warning"
        }

    if action == "get_class_average":
        if not students:
            return {"error": "No student records found."}

        percentages = [
            calculate_attendance(student)[2]
            for student in students
        ]

        return {
            "class_average_attendance":
                f"{sum(percentages) / len(percentages):.1f}%",
            "total_students": len(students)
        }

    if action == "update_attendance":
        roll = command.get("roll") or command.get("roll_no")
        day_value = str(command.get("day", "")).strip()
        status = str(command.get("status", "")).strip().casefold()

        student = find_student(students, roll)

        if student is None:
            return {"error": f"Student with roll number {roll} not found."}

        # Read Day 3, day3, or 3
        day_text = day_value.casefold()

        if day_text.startswith("day"):
            day_text = day_text[3:].strip()

        if not day_text.isdigit():
            return {"error": "Invalid day. Enter Day 1 to Day 5."}

        day_num = int(day_text)
        attendance = student.get("attendance_5_days", [])

        # Ensure the data has the expected five-day list
        if not isinstance(attendance, list):
            return {"error": "attendance_5_days must be a list."}

        if not 1 <= day_num <= len(attendance):
            return {
                "error": (
                    f"Invalid day {day_num}. "
                    f"Valid range is Day 1 to Day {len(attendance)}."
                )
            }

        if status == "present":
            attendance[day_num - 1] = "Present"
        elif status == "absent":
            attendance[day_num - 1] = "Absent"
        elif status in ("unrecorded", "remove", "removed"):
            attendance[day_num - 1] = None
        else:
            return {
                "error": "Status must be Present, Absent, or Unrecorded."
            }

        student["attendance_5_days"] = attendance

        try:
            save_data(students)
        except OSError as error:
            return {"error": f"Could not save attendance: {error}"}

        return {
            "message": "Attendance updated successfully.",
            "student": student
        }

    return {"error": "Unknown action."}


# ---------- OpenRouter AI ----------

def interpret_instruction(user_instruction):
    if not API_KEY:
        return {
            "error": "OPENROUTER_API_KEY is missing from .env."
        }

    system_prompt = """
You are an AI attendance assistant.

Convert the user's instruction into one valid JSON object.
Python performs all calculations and file updates.

Actions:
1. get_student: requires roll
2. get_all_attendance: no extra fields
3. get_class_average: no extra fields
4. update_attendance: requires roll, day, status

Statuses: Present, Absent, Unrecorded.
Use Unrecorded to remove attendance.

Valid student roll numbers are "001", "002", "003", "004", "005".
Preserve leading zeros when possible.
Never invent a roll number or day.
If information is missing, use clarify.
Return JSON only, without Markdown.

Examples:
{"action":"get_student","roll":"001"}
{"action":"get_all_attendance"}
{"action":"get_class_average"}
{"action":"update_attendance","roll":"002","day":"Day 3","status":"Present"}
{"action":"update_attendance","roll":"004","day":"Day 2","status":"Unrecorded"}
{"action":"clarify","question":"Which student's roll number do you mean?"}
"""

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_instruction}
                ],
                "max_tokens": 600,
                "temperature": 0
            },
            timeout=60
        )

        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])

        if not choices:
            return {"error": "AI returned no response."}

        content = choices[0].get("message", {}).get("content")

        if not isinstance(content, str) or not content.strip():
            return {"error": "AI returned an empty response."}

        content = content.strip()

        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        command = json.loads(content)

        if not isinstance(command, dict):
            return {"error": "AI response must be a JSON object."}

        return command

    except requests.exceptions.HTTPError as error:
        details = error.response.text[:500] if error.response else str(error)
        return {"error": f"OpenRouter HTTP error: {details}"}

    except requests.exceptions.RequestException as error:
        return {"error": f"Network/API error: {error}"}

    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return {"error": f"Invalid AI response: {error}"}


# ---------- Main Program ----------

if __name__ == "__main__":
    print("=== AI Attendance Management System ===")
    print("Examples:")
    print("Show attendance of roll number 001")
    print("Show all attendance")
    print("Calculate class average")
    print("Mark attendance of roll number 002 present on Day 3")
    print("Mark roll number 004 absent on Day 2")
    print("Remove attendance of roll number 004 on Day 2")
    print("Type exit to close.")

    while True:
        try:
            prompt = input("\nTerminal Prompt > ").strip()

            if prompt.casefold() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not prompt:
                continue

            command = interpret_instruction(prompt)

            print("\nAI Command:", json.dumps(command, ensure_ascii=False))

            if "error" in command:
                result = command
            else:
                result = execute_action(command)

            print("\n[Execution Result]")
            print(json.dumps(result, indent=4, ensure_ascii=False))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break