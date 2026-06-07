import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from engine import IncidentEngine, ISSUE_CATEGORIES, INTERACTIVE_FLOWS
from database import load_incidents, get_incident, add_incident

app = FastAPI(title="Incident Response Agent")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates = Jinja2Templates(directory=templates_dir)
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

engine = IncidentEngine()


class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    step_worked: bool = None


class SolutionFeedback(BaseModel):
    incident_id: int
    solution_index: int
    success: bool


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        return JSONResponse({"response": "Please describe your issue."})

    sid = req.session_id or str(uuid.uuid4())
    session = engine.get_session(sid)
    is_followup = session is not None and not session.is_done()

    if is_followup and req.step_worked is not None:
        session.advance(req.step_worked)
        if session.resolved:
            s = engine.end_session(sid)
            return JSONResponse({
                "response": "**✅ Issue Resolved!** I'm glad that worked!",
                "session_id": sid,
                "interactive": False,
                "resolved": True,
                "session_summary": s,
            })
        if session.is_done():
            return get_escalation_response(sid, msg, session)

        step = session.current()
        if step is None or step.get("question") is None:
            return get_escalation_response(sid, msg, session)

        diag = None
        if step.get("auto_fix"):
            diag = engine.run_auto_fix(step["auto_fix"])

        return JSONResponse({
            "response": f"**{session.get_progress()['current']}/{session.get_progress()['total']}** - {step['question']}",
            "session_id": sid,
            "interactive": True,
            "step_id": step["id"],
            "auto_fix": step.get("auto_fix"),
            "diagnostic": diag,
            "progress": session.get_progress(),
        })

    classification = engine.classify_issue(msg)
    similar = engine.find_similar_incidents(msg, top_k=5)
    groups = engine.group_incidents()

    if classification["matched"] and INTERACTIVE_FLOWS.get(classification["issue_type"]):
        info = classification["info"]
        session = engine.start_session(sid, classification["issue_type"], msg)
        step = session.current()
        if step is None or step.get("question") is None:
            return get_escalation_response(sid, msg, session)

        diag = None
        if step.get("auto_fix"):
            diag = engine.run_auto_fix(step["auto_fix"])

        root_causes = engine.get_root_cause_probabilities(classification["issue_type"])
        common_fixes = engine.get_common_fixes(classification["issue_type"])

        sim_info = ""
        if similar:
            sim_info = f"\n\n📋 **{len(similar)} similar past incidents found** - will reference if needed."

        causes_html = ""
        if root_causes:
            causes_html = "\n\n**🎯 Root Cause Probability Analysis:**"
            for rc in root_causes:
                bar = "█" * int(rc["probability"] * 20) + "░" * (20 - int(rc["probability"] * 20))
                causes_html += f"\n{bar} {int(rc['probability']*100)}% - {rc['cause']} → *{rc['fix']}*"

        diag_html = ""
        if diag:
            icon = "✅" if diag["status"] == "success" else "⚠️"
            diag_html = f"\n\n**🔍 Auto-Diagnostic:** {icon} {diag['fix_name']}: {diag.get('output', diag.get('message', ''))[:150]}"

        intro = (
            f"**{info['icon']} Identified: {info['label']}** | Severity: {info['severity'].upper()} | Complexity: {info['complexity'].upper()}\n\n"
            f"{session.greeting}{sim_info}{causes_html}{diag_html}"
            f"\n\n**{session.get_progress()['current']}/{session.get_progress()['total']}** - {step['question']}"
        )

        quick_fixes = ""
        if common_fixes:
            quick_fixes = "\n\n**⚡ Quick Fixes you can try right now:**\n" + "\n".join(f"• {f}" for f in common_fixes[:3])

        return JSONResponse({
            "response": intro + quick_fixes,
            "session_id": sid,
            "interactive": True,
            "step_id": step["id"],
            "auto_fix": step.get("auto_fix"),
            "diagnostic": diag,
            "progress": session.get_progress(),
            "issue_info": {"label": info["label"], "icon": info["icon"], "severity": info["severity"], "complexity": info["complexity"]},
            "root_causes": root_causes,
            "common_fixes": common_fixes,
            "similar_incidents": [{"id": m["incident"]["id"], "title": m["incident"]["title"], "similarity": m["similarity"], "severity": m["incident"]["severity"], "affected_users": m["incident"]["affected_users"], "resolve_time": m["incident"]["resolve_time_minutes"]} for m in similar],
        })

    session = engine.start_session(sid, "system_generic", msg)
    step = session.current()
    if step is None or step.get("question") is None:
        return get_escalation_response(sid, msg, session)

    diag = None
    if step.get("auto_fix"):
        diag = engine.run_auto_fix(step["auto_fix"])

    root_causes = engine.get_root_cause_probabilities("system_generic")
    common_fixes = engine.get_common_fixes("system_generic")

    sim_info = ""
    if similar:
        sim_info = "\n\n**📋 Past incidents found:** I'll show you how similar issues were resolved."
        sim_info += "\n\n**Top matches:**"
        for m in similar[:3]:
            inc = m["incident"]
            sim_info += f"\n• **{inc['title']}** ({m['similarity']*100:.0f}%) - {inc['resolution'][:100]}..."

    diag_html = ""
    if diag:
        icon = "✅" if diag["status"] == "success" else "⚠️"
        diag_html = f"\n\n**🔍 Quick Diagnostic:** {icon} {diag['fix_name']}: {diag.get('output', diag.get('message', ''))[:150]}"

    intro = (
        f"**🔧 Analyzing: {msg[:60]}...**\n\n"
        f"{session.greeting}{sim_info}{diag_html}"
        f"\n\n**{session.get_progress()['current']}/{session.get_progress()['total']}** - {step['question']}"
    )
    quick_fixes = ""
    if common_fixes:
        quick_fixes = "\n\n**⚡ Things to try:**\n" + "\n".join(f"• {f}" for f in common_fixes[:4])

    return JSONResponse({
        "response": intro + quick_fixes,
        "session_id": sid,
        "interactive": True,
        "step_id": step["id"],
        "auto_fix": step.get("auto_fix"),
        "diagnostic": diag,
        "progress": session.get_progress(),
        "root_causes": root_causes,
        "common_fixes": common_fixes,
        "similar_incidents": [{"id": m["incident"]["id"], "title": m["incident"]["title"], "similarity": m["similarity"], "severity": m["incident"]["severity"], "affected_users": m["incident"]["affected_users"], "resolve_time": m["incident"]["resolve_time_minutes"]} for m in similar],
    })


def get_escalation_response(sid, msg, session):
    session.escalated = True
    s = engine.end_session(sid)
    similar = engine.find_similar_incidents(msg, top_k=3)

    parts = ["**❌ Could not auto-resolve.** Your issue needs admin attention.\n\n**Steps attempted:**"]
    for r in session.results:
        parts.append(f"{'✅' if r['worked'] else '❌'} {r['step']}: {'Worked' if r['worked'] else 'Failed'}")

    if similar:
        inc = similar[0]["incident"]
        parts.append(f"\n**📖 From our history (when '{inc['title']}' occurred):**\nRoot Cause: {inc['root_cause']}\nResolution: {inc['resolution']}")
        refs = engine.get_external_references(inc)
        if refs:
            parts.append("\n**📚 Helpful resources:**\n" + "\n".join(f"• {r['source']}: {r['title']}" for r in refs))

    parts.append("\n**🚀 Recommended:** Escalate to IT/admin with the diagnostic info above.\n\nYou can also click '🔍 Quick Scan' in the header to run a full system check.")
    return JSONResponse({"response": "\n".join(parts), "session_id": sid, "interactive": False, "resolved": False, "session_summary": s})


@app.post("/api/autofix")
async def autofix(req: dict):
    fix_id = req.get("fix_id")
    if not fix_id:
        return JSONResponse({"error": "No fix specified"})
    result = engine.run_auto_fix(fix_id)
    return JSONResponse(result)


@app.post("/api/scan")
async def system_scan():
    results = engine.run_system_scan()
    status = all(r.get("status") == "success" for r in results.values())
    return JSONResponse({"status": "healthy" if status else "issues_found", "checks": results})


@app.post("/api/feedback")
async def feedback(fb: SolutionFeedback):
    engine.record_solution_outcome(fb.incident_id, fb.solution_index, fb.success)
    return JSONResponse({"status": "ok"})


@app.get("/api/incidents")
async def get_incidents():
    return JSONResponse([{"id": inc["id"], "title": inc["title"], "severity": inc["severity"], "affected_users": inc["affected_users"], "resolve_time_minutes": inc["resolve_time_minutes"], "timestamp": inc["timestamp"], "company": inc["company"], "tags": inc.get("tags", []), "status": inc.get("status", "resolved")} for inc in load_incidents()])


@app.get("/api/incidents/{incident_id}")
async def get_incident_detail(incident_id: int):
    inc = get_incident(incident_id)
    if not inc:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"incident": inc, "references": engine.get_external_references(inc), "recommendations": engine.get_prevention_recommendations(inc)})


@app.get("/api/stats")
async def get_stats():
    incidents = load_incidents()
    total = len(incidents)
    avg = sum(i["resolve_time_minutes"] for i in incidents) / total if total else 0
    sv = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for i in incidents:
        sv[i["severity"]] = sv.get(i["severity"], 0) + 1
    tl = sorted([(i["timestamp"], i["resolve_time_minutes"], i["title"]) for i in incidents])
    return JSONResponse({
        "total_incidents": total, "avg_resolve_time_minutes": round(avg, 1),
        "total_affected_users": sum(i["affected_users"] for i in incidents),
        "severity_distribution": sv,
        "resolve_timeline": [{"date": t[:10], "time": r, "title": title} for t, r, title in tl],
        "total_issue_categories": len(ISSUE_CATEGORIES),
        "auto_fixable_issues": sum(1 for c in ISSUE_CATEGORIES.values() if c.get("auto_fixable")),
    })


@app.get("/api/groups")
async def get_groups():
    return JSONResponse(engine.group_incidents())


@app.get("/api/categories")
async def get_categories():
    return JSONResponse({k: {"label": v["label"], "icon": v["icon"], "severity": v["severity"], "complexity": v["complexity"], "auto_fixable": v["auto_fixable"]} for k, v in ISSUE_CATEGORIES.items()})


@app.get("/api/session-history")
async def session_history():
    return JSONResponse(engine.get_session_history())


@app.post("/api/incidents")
async def create_incident(incident: dict):
    new_inc = add_incident(incident)
    engine.refresh()
    return JSONResponse(new_inc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
