import os
import io
import json
from datetime import datetime

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="VA ClaimMate MVP", layout="wide")

client = OpenAI()

DATA_FILE = "claimmate_data.json"


def load_app_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    defaults = {
        "veteran_profile": {},
        "issues": [],
        "symptom_mappings": [],
        "symptom_note": "",
        "claims": [],
        "notes": "",
        "documents": [],
        "evidence_summary": "",
    }
    for k, v in defaults.items():
        if k not in data:
            data[k] = v
    return data


def save_app_data(data):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Save error: {e}")


def get_state():
    if "app_data" not in st.session_state:
        st.session_state.app_data = load_app_data()
    return st.session_state.app_data


def persist_state():
    if "app_data" in st.session_state:
        save_app_data(st.session_state.app_data)


def call_gpt(system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.3):
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Model error: {e}")
        return ""


def extract_text_from_bytes(content: bytes, mime: str, name: str) -> str:
    text = ""
    mime = mime or ""

    if mime.startswith("text/"):
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    elif mime == "application/pdf":
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            text = "\n".join(pages)
        except Exception as e:
            st.warning(f"PDF reading failed for {name}: {e}")
            text = ""
    elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            from docx import Document

            document = Document(io.BytesIO(content))
            paras = [p.text for p in document.paragraphs]
            text = "\n".join(paras)
        except Exception as e:
            st.warning(f"DOCX reading failed for {name}: {e}")
            text = ""
    else:
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    return text


def map_symptoms(symptom_text: str):
    system_prompt = (
        "You support veterans building VA disability claims. "
        'Return JSON only. Format: '
        '[{"condition":"","icd10":"","body_system":"","va_rating_hint":"","rationale":""}] '
        "Do not add commentary outside JSON."
    )

    user_prompt = (
        "Symptoms and history from the veteran:\n"
        f"{symptom_text}\n\n"
        "Suggest likely diagnostic labels with ICD-10 codes and VA rating hints. "
        "Include Gulf War environmental exposure links when that fits."
    )

    raw = call_gpt(system_prompt, user_prompt, temperature=0.2)
    parsed = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            parsed = data
    except Exception:
        parsed = []

    return parsed, raw


def build_personal_statement(app_data, claim_title: str, focus_conditions):
    prof = app_data.get("veteran_profile", {})
    issues = app_data.get("issues", [])
    symptom_mappings = app_data.get("symptom_mappings", [])
    evidence_summary = app_data.get("evidence_summary", "")

    prof_lines = []
    if prof.get("full_name"):
        prof_lines.append(f"Name: {prof.get('full_name')}")
    if prof.get("branch"):
        prof_lines.append(f"Branch: {prof.get('branch')}")
    if prof.get("service_dates"):
        prof_lines.append(f"Service dates: {prof.get('service_dates')}")
    if prof.get("deployment_locations"):
        prof_lines.append(f"Deployments: {prof.get('deployment_locations')}")
    if prof.get("mos_duties"):
        prof_lines.append(f"Duties: {prof.get('mos_duties')}")
    if prof.get("other_notes"):
        prof_lines.append(f"Other context: {prof.get('other_notes')}")

    prof_block = "\n".join(prof_lines)

    issue_lines = [f"- {i.get('label')}" for i in issues if i.get("label")]
    issues_block = "\n".join(issue_lines)

    mapping_lines = []
    for m in symptom_mappings:
        if focus_conditions and m.get("condition") not in focus_conditions:
            continue
        mapping_lines.append(
            f"- {m.get('condition')} (ICD-10 {m.get('icd10')}, system: {m.get('body_system')}) "
            f"Hint: {m.get('va_rating_hint')}"
        )
    mappings_block = "\n".join(mapping_lines)

    system_prompt = (
        "You write VA lay statements and personal impact statements for disability claims. "
        "Use first person from the veteran. "
        "Tone: plain language, honest, detailed, no legal jargon. "
        "Target length: 600 to 900 words. "
        "Cover onset, progression, daily impact, work impact, safety concerns, sleep, mental health, flare patterns, "
        "and connection to service. "
        "Reference medical and symptom context when helpful, without quoting records line by line."
    )

    user_prompt = (
        f"Claim focus title:\n{claim_title}\n\n"
        f"Veteran background:\n{prof_block}\n\n"
        f"High level claimed issues:\n{issues_block}\n\n"
        f"Selected conditions and VA rating hints:\n{mappings_block}\n\n"
        f"Evidence summary prepared from records:\n{evidence_summary}\n\n"
        "Write a lay statement for this claim. "
        "Use first person, talk through a typical day, and explain functional impact. "
        "End with a short paragraph thanking the rater and confirming that the statement is true to the best of the veteran's knowledge."
    )

    return call_gpt(system_prompt, user_prompt, temperature=0.35)


def build_full_claim_packet(app_data):
    prof = app_data.get("veteran_profile", {})
    issues = app_data.get("issues", [])
    symptom_mappings = app_data.get("symptom_mappings", [])
    claims = app_data.get("claims", [])
    evidence_summary = app_data.get("evidence_summary", "")

    lines = []

    lines.append("VA ClaimMate Claim Packet")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")

    lines.append("Veteran profile")
    lines.append("----------------")
    if prof.get("full_name"):
        lines.append(f"Name: {prof.get('full_name')}")
    if prof.get("branch"):
        lines.append(f"Branch: {prof.get('branch')}")
    if prof.get("service_dates"):
        lines.append(f"Service dates: {prof.get('service_dates')}")
    if prof.get("deployment_locations"):
        lines.append(f"Deployments: {prof.get('deployment_locations')}")
    if prof.get("mos_duties"):
        lines.append(f"Duties / MOS: {prof.get('mos_duties')}")
    if prof.get("other_notes"):
        lines.append(f"Additional notes: {prof.get('other_notes')}")
    lines.append("")

    lines.append("Claimed issues")
    lines.append("-------------")
    for i in issues:
        if i.get("label"):
            detail = i.get("details") or ""
            lines.append(f"- {i.get('label')} {(' - ' + detail) if detail else ''}")
    lines.append("")

    lines.append("Symptom to condition mappings")
    lines.append("-----------------------------")
    for m in symptom_mappings:
        lines.append(
            f"- {m.get('condition')} | ICD-10 {m.get('icd10')} | system: {m.get('body_system')} "
            f"| rating hint: {m.get('va_rating_hint')} "
            f"| selected: {m.get('selected_for_claim', False)}"
        )
    lines.append("")

    lines.append("Evidence summary")
    lines.append("----------------")
    lines.append(evidence_summary or "[No summary prepared yet]")
    lines.append("")

    lines.append("Saved personal statements")
    lines.append("-------------------------")
    for c in claims:
        lines.append("")
        lines.append(f"Title: {c.get('title')}")
        lines.append(f"Created: {c.get('created_at')}")
        lines.append("")
        lines.append(c.get("body") or "")
        lines.append("")

    return "\n".join(lines)


def build_chat_context(app_data):
    prof = app_data.get("veteran_profile", {})
    issues = app_data.get("issues", [])
    evidence_summary = app_data.get("evidence_summary", "")
    docs = app_data.get("documents", [])

    prof_bits = []
    if prof.get("branch"):
        prof_bits.append(f"Branch: {prof.get('branch')}")
    if prof.get("service_dates"):
        prof_bits.append(f"Service dates: {prof.get('service_dates')}")
    if prof.get("deployment_locations"):
        prof_bits.append(f"Deployments: {prof.get('deployment_locations')}")

    issues_bits = [i.get("label") for i in issues if i.get("label")]

    doc_snippets = []
    for d in docs:
        text = d.get("text") or ""
        if text:
            snippet = text[:1200]
            doc_snippets.append(f"{d.get('name')}:\n{snippet}")

    pieces = []
    if prof_bits:
        pieces.append("Profile:\n" + "\n".join(prof_bits))
    if issues_bits:
        pieces.append("Claimed issues:\n- " + "\n- ".join(issues_bits))
    if evidence_summary:
        pieces.append("Evidence summary:\n" + evidence_summary)
    if doc_snippets:
        pieces.append("Record snippets:\n" + "\n\n".join(doc_snippets[:3]))

    return "\n\n".join(pieces)


def main():
    app_data = get_state()

    st.title("VA ClaimMate MVP")
    st.caption("Step by step helper for stronger VA disability claim preparation.")

    tabs = st.tabs(
        [
            "1. Profile & Issues",
            "2. Upload Evidence",
            "3. Symptom to Condition Mapper",
            "4. Personal Statement Builder",
            "5. Saved Claims Dashboard",
            "6. General VA Claim Chat",
        ]
    )

    with tabs[0]:
        st.subheader("Step 1. Veteran profile and claimed issues")
        st.write(
            "Use this section to store basic service history and a simple list of claim issues. "
            "Other tabs pull from this data."
        )

        vp = app_data.get("veteran_profile", {})

        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input(
                "Full name",
                value=vp.get("full_name", ""),
            )
            branch = st.text_input(
                "Branch of service",
                value=vp.get("branch", ""),
            )
            service_dates = st.text_input(
                "Service dates (for example: 1989-03 to 1996-01)",
                value=vp.get("service_dates", ""),
            )
            deployment_locations = st.text_area(
                "Deployments / base locations",
                value=vp.get("deployment_locations", ""),
                height=80,
            )

        with col2:
            mos_duties = st.text_area(
                "Duties, AFSC/MOS, and role details",
                value=vp.get("mos_duties", ""),
                height=120,
            )
            other_notes = st.text_area(
                "Other background notes for the rater",
                value=vp.get("other_notes", ""),
                height=120,
            )

        vp["full_name"] = full_name
        vp["branch"] = branch
        vp["service_dates"] = service_dates
        vp["deployment_locations"] = deployment_locations
        vp["mos_duties"] = mos_duties
        vp["other_notes"] = other_notes
        app_data["veteran_profile"] = vp

        st.markdown("---")

        st.markdown("#### Claimed issues list")
        existing_labels = [i.get("label", "") for i in app_data.get("issues", [])]
        issues_text = st.text_area(
            "Enter one issue per line (for example: asthma, cervical spine, PTSD, Gulf War multi-symptom illness)",
            value="\n".join(existing_labels),
            height=140,
        )

        new_issues = []
        for line in issues_text.splitlines():
            line = line.strip()
            if line:
                new_issues.append({"label": line, "details": ""})
        app_data["issues"] = new_issues

    with tabs[1]:
        st.subheader("Step 2. Upload medical records and evidence")
        st.write(
            "Upload VA records, C&P exams, DBQs, private records, or STRs. "
            "The app extracts text for later analysis and summaries."
        )

        uploads = st.file_uploader(
            "Select one or more files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )

        if uploads:
            for up in uploads:
                content = up.getvalue()
                mime = up.type
                name = up.name
                size = len(content)

                existing_ids = [d.get("id") for d in app_data["documents"]]
                doc_id = f"{name}:{size}"
                if doc_id in existing_ids:
                    continue

                text = extract_text_from_bytes(content, mime, name)
                doc_entry = {
                    "id": doc_id,
                    "name": name,
                    "mime": mime,
                    "size": size,
                    "uploaded_at": datetime.utcnow().isoformat() + "Z",
                    "text": text,
                    "notes": "",
                }
                app_data["documents"].append(doc_entry)

        if app_data["documents"]:
            st.markdown("#### Uploaded documents")
            for idx, doc in enumerate(app_data["documents"]):
                with st.expander(f"{doc.get('name')} ({doc.get('mime')}, {doc.get('size')} bytes)"):
                    st.write(f"Uploaded: {doc.get('uploaded_at')}")
                    doc_notes = st.text_area(
                        "Notes for this document",
                        value=doc.get("notes", ""),
                        key=f"doc_notes_{idx}",
                    )
                    doc["notes"] = doc_notes
                    sample = (doc.get("text") or "")[:1200]
                    if sample:
                        st.text_area(
                            "First part of extracted text",
                            value=sample,
                            height=160,
                            key=f"doc_preview_{idx}",
                        )
                    else:
                        st.info("No text extracted.")

        st.markdown("---")
        st.markdown("#### Combined evidence summary")

        current_summary = app_data.get("evidence_summary", "")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Build or refresh evidence summary"):
                joined_texts = []
                for d in app_data["documents"]:
                    text = d.get("text") or ""
                    if text:
                        joined_texts.append(f"{d.get('name')}:\n{text}")
                big_blob = "\n\n".join(joined_texts)
                big_blob = big_blob[:12000]

                if big_blob:
                    system_prompt = (
                        "You summarize medical records for VA disability claims. "
                        "Write a structured summary covering diagnoses, key findings, "
                        "functional limitations, and any references to service or exposures."
                    )
                    user_prompt = (
                        "Summarize the following records for use in a VA claim. "
                        "Do not give legal advice.\n\n"
                        f"{big_blob}"
                    )
                    summary = call_gpt(system_prompt, user_prompt, temperature=0.25)
                    app_data["evidence_summary"] = summary
                    current_summary = summary
        with col_b:
            st.caption("Use the button to let the model read uploaded text and draft a combined overview.")

        evidence_summary_text = st.text_area(
            "Evidence summary (editable)",
            value=current_summary,
            height=220,
        )
        app_data["evidence_summary"] = evidence_summary_text

    with tabs[2]:
        st.subheader("Step 3. Symptom to condition mapper")
        st.write(
            "Describe symptoms, flare patterns, and service exposures. "
            "The model proposes diagnostic labels with ICD-10 codes and rating hints."
        )

        symptom_note = st.text_area(
            "Describe symptoms and history in your own words",
            value=app_data.get("symptom_note", ""),
            height=220,
        )
        app_data["symptom_note"] = symptom_note

        if st.button("Analyze symptoms and suggest conditions"):
            if symptom_note.strip():
                mappings, raw = map_symptoms(symptom_note)
                if mappings:
                    app_data["symptom_mappings"] = mappings
                else:
                    st.warning("No JSON parsed from model response. Raw output shown below.")
                    st.text_area("Raw model output", value=raw, height=200, key="raw_symptom_output")

        if app_data["symptom_mappings"]:
            st.markdown("#### Suggested conditions")

            all_names = [m.get("condition", "") for m in app_data["symptom_mappings"] if m.get("condition")]
            preselected = [
                m.get("condition")
                for m in app_data["symptom_mappings"]
                if m.get("selected_for_claim") and m.get("condition")
            ]

            selected = st.multiselect(
                "Select conditions to link to current claim",
                all_names,
                default=preselected,
            )

            for m in app_data["symptom_mappings"]:
                m["selected_for_claim"] = m.get("condition") in selected

            rows = []
            for m in app_data["symptom_mappings"]:
                rows.append(
                    {
                        "Condition": m.get("condition"),
                        "ICD-10": m.get("icd10"),
                        "Body system": m.get("body_system"),
                        "VA rating hint": m.get("va_rating_hint"),
                        "Selected": m.get("selected_for_claim", False),
                    }
                )
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)

    with tabs[3]:
        st.subheader("Step 4. Personal statement builder")
        st.write(
            "Use profile, symptom mapping, and evidence summary to draft lay statements for each claimed issue."
        )

        issues = app_data.get("issues", [])
        symptom_mappings = app_data.get("symptom_mappings", [])

        condition_names = [
            m.get("condition")
            for m in symptom_mappings
            if m.get("selected_for_claim") and m.get("condition")
        ]

        default_title = ""
        if condition_names:
            default_title = ", ".join(condition_names[:3])
        elif issues:
            default_title = ", ".join(i.get("label") for i in issues[:3] if i.get("label"))

        claim_title = st.text_input(
            "Title or focus for this statement (for example: Asthma and Gulf War respiratory exposure)",
            value=default_title,
        )

        focus_conditions = st.multiselect(
            "Focus on these conditions in this statement",
            condition_names,
            default=condition_names,
        )

        generated_key = "current_generated_statement"
        if generated_key not in st.session_state:
            st.session_state[generated_key] = ""

        if st.button("Generate personal statement"):
            if claim_title.strip():
                text = build_personal_statement(app_data, claim_title, focus_conditions)
                st.session_state[generated_key] = text

        current_statement = st.text_area(
            "Personal statement (editable)",
            value=st.session_state.get(generated_key, ""),
            height=260,
        )
        st.session_state[generated_key] = current_statement

        if st.button("Save statement to claims"):
            if current_statement.strip():
                app_data["claims"].append(
                    {
                        "id": f"claim_{len(app_data['claims']) + 1}_{int(datetime.utcnow().timestamp())}",
                        "title": claim_title or "VA claim statement",
                        "body": current_statement,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    }
                )
                st.success("Statement saved to claims dashboard.")

    with tabs[4]:
        st.subheader("Step 5. Saved claims dashboard and export")
        st.write(
            "Review saved statements and export a combined claim packet for download."
        )

        claims = app_data.get("claims", [])

        if not claims:
            st.info("No saved statements yet.")
        else:
            remove_ids = []
            for c in claims:
                with st.expander(f"{c.get('title')} (created {c.get('created_at')})"):
                    st.text_area(
                        "Statement text",
                        value=c.get("body") or "",
                        height=220,
                        key=f"claim_body_{c.get('id')}",
                    )
                    if st.button("Remove this statement", key=f"remove_{c.get('id')}"):
                        remove_ids.append(c.get("id"))

            if remove_ids:
                app_data["claims"] = [c for c in claims if c.get("id") not in remove_ids]
                st.success("Selected statements removed.")

        st.markdown("---")

        packet_text = build_full_claim_packet(app_data)
        st.download_button(
            label="Download full claim packet as text (.txt)",
            data=packet_text,
            file_name="va_claimmate_packet.txt",
            mime="text/plain",
        )

    with tabs[5]:
        st.subheader("General VA claim chat")
        st.write(
            "Use this chat for general questions. The assistant uses your profile, issues, evidence summary, and record snippets as context."
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_msg = st.chat_input("Ask a question about your VA claim or evidence.")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            context = build_chat_context(app_data)
            system_prompt = (
                "You are a VA claims helper. "
                "Provide education on VA concepts, rating criteria, and preparation steps. "
                "Do not give legal advice and do not promise outcomes. "
                "Use the provided context when relevant, but you are not a substitute for a VSO, attorney, or accredited representative."
            )
            user_prompt = (
                "Context for this veteran:\n"
                f"{context}\n\n"
                f"User question:\n{user_msg}"
            )
            reply = call_gpt(system_prompt, user_prompt, temperature=0.35)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})

            with st.chat_message("assistant"):
                st.write(reply)

    persist_state()


if __name__ == "__main__":
    main()
