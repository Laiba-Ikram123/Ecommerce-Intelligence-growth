import json
import re
import requests

NUM_RE = re.compile(r"(?<![A-Za-z0-9])\d[\d,]*(?:\.\d+)?")
ACTION_KEYS = ["priority", "problem", "evidence", "financial_impact", "suggested_action"]


def build_compact_payload(result):
    """Small, evidence-only summary for the LLM. Full per-customer/per-product tables are NOT sent,
    which keeps the prompt well under the model's token limit even for large datasets."""
    findings = []
    for f in result.get("investigation", []):
        tables = f.get("tables", [])
        findings.append({
            "title": f["title"], "severity": f["severity"], "confidence": f["confidence"],
            "observed": f["observed"],
            "top_evidence": tables[0]["rows"][:3] if tables else [],
            "possible_causes": [{"cause": c["cause"], "strength": c["strength"]} for c in f["possible_causes"][:4]],
            "next_steps": f["next_steps"][:3],
            "data_gaps": f.get("data_gaps", []),
        })
    return {
        "kpis": result["kpis"],
        "agent_highlights": {a["name"]: a["highlights"] for a in result["agents"]},
        "rule_based_actions": result["orchestrator"].get("actions", []),
        "investigation": findings,
        "data_warnings": result.get("health", {}).get("warnings", []),
    }


def _numbers(text):
    out = []
    for m in NUM_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def verify_figures(ai, payload):
    """Guardrail: every number the LLM wrote must match a number Python computed (allowing rounding).
    This checks the figures, not the reasoning."""
    allowed = set()
    for v in _numbers(json.dumps(payload, default=str)):
        allowed.update({round(v, 2), round(v, 1), float(round(v))})
    checked, unmatched = 0, []
    for a in ai.get("actions", []):
        for key in ACTION_KEYS:
            for v in _numbers(str(a.get(key, ""))):
                checked += 1
                if not ({round(v, 2), round(v, 1), float(round(v))} & allowed):
                    unmatched.append(f"{v:g}")
    return {"checked": checked, "unmatched": sorted(set(unmatched))}


def generate_ai_insights(result, api_key):
    payload = build_compact_payload(result)
    prompt = """You are the Orchestrator of an e-commerce business intelligence system.
You receive outputs from five agents (Sales, Product, Customer, Returns, Financial) and an Investigation layer
that already explains WHY things happened and proposes next steps. All numbers were computed by Python.

Rules:
- Use ONLY facts and numbers present in the data below. Copy figures exactly as given; do not calculate new numbers.
- Distinguish observed evidence from possible explanations. Customer-reported reasons and segment patterns are
  associations, not proof. Respect each finding's confidence level; say so when confidence is Low.
- Combine signals ACROSS agents (e.g. a product that is both high-return and low-margin) rather than repeating one agent.
- Mention data gaps when they limit the conclusion.

Return JSON only with this structure:
{"actions":[{"priority":"High/Medium/Low","problem":"...","evidence":"...","financial_impact":"...","suggested_action":"..."}]}
Create 2-4 cross-agent business actions.

DATA:
""" + json.dumps(payload, default=str)

    # Groq OpenAI-compatible endpoint
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        },
        timeout=45
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    ai = json.loads(content)

    actions = ai.get("actions") if isinstance(ai, dict) else None
    if not isinstance(actions, list) or not actions:
        raise ValueError("The model did not return any actions.")
    clean = []
    for a in actions:
        if isinstance(a, dict):
            clean.append({k: str(a.get(k, "-")) for k in ACTION_KEYS})
    if not clean:
        raise ValueError("The model's actions were not in the expected format.")
    ai = {"actions": clean}
    ai["verification"] = verify_figures(ai, payload)
    return ai
