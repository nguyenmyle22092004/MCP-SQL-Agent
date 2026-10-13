from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "traces"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class TraceRecorder:
    def __init__(self, user_id: str = "unknown", conversation_id: str = "unknown", message: str = ""):
        self.trace_id = f"trace_{uuid4().hex[:16]}"
        self.data = {
            "trace_id": self.trace_id,
            "request": {"user_id": user_id, "conversation_id": conversation_id, "message": message},
            "events": [],
            "llm_calls": [],
            "tool_calls": [],
            "evidence": [],
            "claims": [],
            "validation": {"status": "pending", "errors": []},
        }

    def add_event(self, event_type: str, **payload):
        self.data["events"].append({"event_type": event_type, **payload})

    def start_llm_call(self, model: str, purpose: str, **payload) -> str:
        call_id = f"llm_{uuid4().hex[:12]}"
        record = {
            "llm_call_id": call_id,
            "model": model,
            "purpose": purpose,
            "status": "running",
        }
        self.data["llm_calls"].append(record)
        self.add_event("LLM_CALL_START", llm_call_id=call_id, model=model, purpose=purpose, **payload)
        return call_id

    def end_llm_call(self, call_id: str, status: str, **payload) -> None:
        record = self._find_record(self.data["llm_calls"], "llm_call_id", call_id)
        if record is not None:
            record.update({"status": status, **payload})
        self.add_event("LLM_CALL_END", llm_call_id=call_id, status=status, **payload)

    def record_decision(self, llm_call_id: str | None, decision: str, reason: str, **payload) -> str:
        decision_id = f"decision_{uuid4().hex[:12]}"
        self.add_event(
            "AGENT_DECISION",
            decision_id=decision_id,
            llm_call_id=llm_call_id,
            decision=decision,
            reason=reason,
            **payload,
        )
        return decision_id

    def record_policy_check(
        self,
        decision_id: str | None,
        policy: str,
        status: str,
        reason: str = "",
        **payload,
    ) -> str:
        policy_check_id = f"policy_{uuid4().hex[:12]}"
        self.add_event(
            "POLICY_CHECK",
            policy_check_id=policy_check_id,
            decision_id=decision_id,
            policy=policy,
            status=status,
            reason=reason,
            **payload,
        )
        return policy_check_id

    def add_tool_call(
        self,
        server: str,
        tool: str,
        payload: dict | None = None,
        tool_call_id: str | None = None,
        decision_id: str | None = None,
    ):
        tool_call_id = tool_call_id or f"tool_{uuid4().hex[:12]}"
        record = {
            "server": server,
            "tool": tool,
            "tool_call_id": tool_call_id,
            "decision_id": decision_id,
            "payload": payload or {},
            "status": "running",
        }
        self.data["tool_calls"].append(record)
        self.add_event(
            "MCP_CALL_START",
            tool_call_id=tool_call_id,
            decision_id=decision_id,
            server=server,
            tool=tool,
            request=payload or {},
        )
        return tool_call_id

    def end_tool_call(self, tool_call_id: str, status: str, **payload) -> None:
        record = self._find_record(self.data["tool_calls"], "tool_call_id", tool_call_id)
        if record is not None:
            record.update({"status": status, **payload})
        self.add_event("MCP_CALL_END", tool_call_id=tool_call_id, status=status, **payload)

    def record_observation(self, tool_call_id: str, observation: dict, source: str = "mcp") -> str:
        observation_id = f"obs_{uuid4().hex[:12]}"
        self.add_event(
            "TOOL_OBSERVATION",
            observation_id=observation_id,
            tool_call_id=tool_call_id,
            source=source,
            observation=observation,
        )
        return observation_id

    def add_evidence(self, evidence: dict):
        evidence_record = dict(evidence)
        evidence_record.setdefault("evidence_id", f"ev_{uuid4().hex[:12]}")
        self.data["evidence"].append(evidence_record)
        self.add_event("EVIDENCE_CREATED", evidence_id=evidence_record["evidence_id"], evidence=evidence_record)
        return evidence_record["evidence_id"]

    def add_claim(self, claim: str | dict):
        self.data["claims"].append(claim)

    def set_validation(self, status: str, errors: list[str] | None = None):
        self.data["validation"] = {"status": status, "errors": errors or []}

    @staticmethod
    def _find_record(records: list[dict], key: str, value: str) -> dict | None:
        return next((record for record in records if record.get(key) == value), None)

    def save(self):
        errors = self.validate_contract()
        self.data["trace_contract"] = {"status": "failed" if errors else "pass", "errors": errors}
        self.add_event("TRACE_CONTRACT_CHECK", status=self.data["trace_contract"]["status"], errors=errors)
        if errors and self.data["validation"].get("status") == "pass":
            self.set_validation("failed", [*self.data["validation"].get("errors", []), *errors])
        path = LOG_DIR / f"{self.trace_id}.json"
        path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return self.trace_id

    def validate_contract(self) -> list[str]:
        errors = []
        events = self.data["events"]

        def matching_events(event_type: str, key: str, value: str) -> list[dict]:
            return [event for event in events if event.get("event_type") == event_type and event.get(key) == value]

        llm_records = self.data["llm_calls"]
        llm_call_ids = {record.get("llm_call_id") for record in llm_records}
        if len(llm_call_ids) != len(llm_records):
            errors.append("LLM call IDs must be unique.")
        for record in llm_records:
            call_id = record.get("llm_call_id")
            starts = matching_events("LLM_CALL_START", "llm_call_id", call_id)
            ends = matching_events("LLM_CALL_END", "llm_call_id", call_id)
            if len(starts) != 1:
                errors.append(f"LLM call {call_id} must have exactly one start event.")
            if len(ends) != 1 or record.get("status") == "running":
                errors.append(f"LLM call {call_id} has no end event or has duplicate end events.")
            elif ends[0].get("status") != record.get("status"):
                errors.append(f"LLM call {call_id} end-event status does not match its record.")
        for event_type in ("LLM_CALL_START", "LLM_CALL_END"):
            for event in (item for item in events if item.get("event_type") == event_type):
                if event.get("llm_call_id") not in llm_call_ids:
                    errors.append(f"{event_type} references a missing LLM call.")

        tool_records = self.data["tool_calls"]
        tool_call_ids = {record.get("tool_call_id") for record in tool_records}
        if len(tool_call_ids) != len(tool_records):
            errors.append("Tool call IDs must be unique.")
        decision_events = [event for event in events if event.get("event_type") == "AGENT_DECISION"]
        decision_ids = {event.get("decision_id") for event in decision_events}
        if len(decision_ids) != len(decision_events):
            errors.append("Decision IDs must be unique.")
        for event in decision_events:
            llm_call_id = event.get("llm_call_id")
            if llm_call_id and llm_call_id not in llm_call_ids:
                errors.append(f"Decision {event.get('decision_id')} references a missing LLM call.")
        policy_events = [event for event in events if event.get("event_type") == "POLICY_CHECK"]
        policy_check_ids = {event.get("policy_check_id") for event in policy_events}
        if len(policy_check_ids) != len(policy_events):
            errors.append("Policy check IDs must be unique.")
        for event in policy_events:
            if event.get("decision_id") not in decision_ids:
                errors.append(f"Policy check {event.get('policy_check_id')} references a missing decision.")
        for decision_id in decision_ids:
            if not any(event.get("decision_id") == decision_id for event in policy_events):
                errors.append(f"Decision {decision_id} has no policy check.")
        for record in tool_records:
            call_id = record.get("tool_call_id")
            starts = matching_events("MCP_CALL_START", "tool_call_id", call_id)
            ends = matching_events("MCP_CALL_END", "tool_call_id", call_id)
            if len(starts) != 1:
                errors.append(f"Tool call {call_id} must have exactly one start event.")
            elif any(
                starts[0].get(key) != value
                for key, value in (("server", record.get("server")), ("tool", record.get("tool")), ("decision_id", record.get("decision_id")), ("request", record.get("payload", {})))
            ):
                errors.append(f"Tool call {call_id} start event does not match its request record.")
            if len(ends) != 1 or record.get("status") == "running":
                errors.append(f"Tool call {call_id} has no end event or has duplicate end events.")
            elif ends[0].get("status") != record.get("status"):
                errors.append(f"Tool call {call_id} end-event status does not match its record.")
            decision_id = record.get("decision_id")
            if not decision_id or decision_id not in decision_ids:
                errors.append(f"Tool call {call_id} has no valid agent decision.")
            elif not any(event.get("decision_id") == decision_id and event.get("status") == "pass" for event in policy_events):
                errors.append(f"Tool call {call_id} has no passing policy check for its decision.")
            observations = matching_events("TOOL_OBSERVATION", "tool_call_id", call_id)
            if len(observations) != 1:
                errors.append(f"Tool call {call_id} has no observation or has duplicate observations.")
        observation_ids = {
            event.get("observation_id") for event in events if event.get("event_type") == "TOOL_OBSERVATION"
        }
        observation_events = [event for event in events if event.get("event_type") == "TOOL_OBSERVATION"]
        if len(observation_ids) != len(observation_events):
            errors.append("Observation IDs must be unique.")
        for event_type in ("MCP_CALL_START", "MCP_CALL_END", "TOOL_OBSERVATION"):
            for event in (item for item in events if item.get("event_type") == event_type):
                if event.get("tool_call_id") not in tool_call_ids:
                    errors.append(f"{event_type} references a missing tool call.")
        evidence_records = self.data["evidence"]
        evidence_ids = {record.get("evidence_id") for record in evidence_records}
        if len(evidence_ids) != len(evidence_records):
            errors.append("Evidence IDs must be unique.")
        for evidence in self.data["evidence"]:
            evidence_id = evidence.get("evidence_id")
            if evidence.get("trace_id") != self.trace_id:
                errors.append(f"Evidence {evidence_id} belongs to a different trace.")
            if evidence.get("tool_call_id") not in tool_call_ids:
                errors.append(f"Evidence {evidence_id} references a missing tool call.")
            evidence_events = [
                event for event in events
                if event.get("event_type") == "EVIDENCE_CREATED" and event.get("evidence_id") == evidence_id
            ]
            if len(evidence_events) != 1:
                errors.append(f"Evidence {evidence_id} must have exactly one creation event.")
            elif evidence_events[0].get("evidence", {}).get("tool_call_id") != evidence.get("tool_call_id"):
                errors.append(f"Evidence {evidence_id} creation event is linked to a different tool call.")
        for event in (item for item in events if item.get("event_type") == "EVIDENCE_CREATED"):
            if event.get("evidence_id") not in evidence_ids:
                errors.append("Evidence creation event references a missing evidence record.")
        return errors
