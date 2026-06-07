import json
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "incidents.json")


def load_incidents():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_incidents(incidents):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2, ensure_ascii=False)


def get_incident(incident_id: int) -> Optional[dict]:
    incidents = load_incidents()
    for inc in incidents:
        if inc["id"] == incident_id:
            return inc
    return None


def add_incident(incident: dict) -> dict:
    incidents = load_incidents()
    new_id = max(i["id"] for i in incidents) + 1 if incidents else 1
    incident["id"] = new_id
    incident["status"] = "open"
    if "solutions" not in incident:
        incident["solutions"] = []
    incidents.append(incident)
    save_incidents(incidents)
    return incident


def update_solution_success(incident_id: int, solution_index: int, success: bool):
    incidents = load_incidents()
    for inc in incidents:
        if inc["id"] == incident_id:
            sols = inc.get("solutions", [])
            if solution_index < len(sols):
                sols[solution_index]["times_used"] += 1
                if success:
                    sols[solution_index]["times_successful"] += 1
                sols[solution_index]["success_rate"] = round(
                    sols[solution_index]["times_successful"] / sols[solution_index]["times_used"], 2
                )
    save_incidents(incidents)
