import os
import io
import sqlite3
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
import docx

# --- Configuration ---
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

# --- Database setup ---
DB_PATH = "claims.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

def setup_database():
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            user TEXT NOT NULL,
            conditions TEXT,
            statement TEXT,
            evidence_summary TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user TEXT NOT NULL,
            content TEXT NOT NULL
        )
        """
    )
    conn.commit()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username: str, password: str) -> str:
    if not username or not password:
        return "Username and password cannot be empty."
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_password(password)),
        )
        conn.commit()
        return "Account created. You can log in."
    except sqlite3.IntegrityError:
        return "Username already exists. Pick another."
    except Exception as e:
        return f"Error: {e}"

def authenticate_user(username: str, password: str) -> bool:
    if not username or not password:
        return False
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if not row:
        return False
    return row[0] == hash_password(password)

# --- Evidence text extraction ---
def extract_text(file) -> str:
    try:
        bytes_data = file.read()
        file_stream = io.BytesIO(bytes_data)
        name = file.name.lower()

        if name.endswith(".pdf"):
            doc = fitz.open(stream=file_stream, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text

        if name.endswith(".docx"):
            document = docx.Document(file_stream)
            return "\n".join(p.text for p in document.paragraphs)

        return "Unsupported file type. Please upload a PDF or DOCX."
    except Exception as e:
        return f"Error extracting text: {e}"

# --- AI helpers ---
def require_client():
    if client is None:
        return "OpenAI API key is not set in OPENAI_API_KEY."
    return None

def generate_statement(service_branch, service_years, conditions, symptom_description, name, file_number, user):
    error = require_client()
    if error:
        return error

    if not all([service_branch, service_years, conditions, symptom_description, name, file_number]):
        return "Fill in all fields first."

    prompt = (
        f"I am a U.S. military veteran who served in the {service_branch} for {service_years} years. "
        f"I am filing a VA disability claim for the following conditions: {conditions}. "
        f"My symptoms and their impact on my daily life are as follows: {symptom_description}. "
        "Write a strong, personal VA statement in the first person that clearly links my service to my conditions. "
        "Keep the tone authentic and respectful. Conclude with my name and VA file number.\n\n"
        f"Full Name: {name}\nVA File Number: {file_number}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You write clear, empathetic, persuasive personal statements for VA disability claims.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        statement = response.choices[0].message.content

        cursor = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO claims (date, user, conditions, statement) VALUES (?, ?, ?, ?)",
            (date, user, conditions, statement),
        )
        conn.commit()

        return statement
    except Exception as e:
        return f"Error generating statement: {e}"

def rewrite_statement(existing_statement: str) -> str:
    error = require_client()
    if error:
        return error

    if not existing_statement:
        return "No statement to rewrite."

    prompt = (
        "Improve this VA disability claim statement. "
        "Keep the same story and tone, but make it clearer and more persuasive. "
        "Ensure the link between service and conditions is obvious.\n\n"
        f"{existing_statement}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert editor for VA disability claim statements.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error rewriting statement: {e}"

def map_symptoms_to_conditions(symptoms: str):
    error = require_client()
    if error:
        return pd.DataFrame({"Error": [error]})

    if not symptoms:
        return pd.DataFrame({"Message": ["Enter symptoms first."]})

    prompt = (
        f"A U.S. veteran lists these symptoms for a VA disability claim: '{symptoms}'. "
        "List the most likely VA-claimable medical conditions with their ICD-10 codes. "
        "Respond as a markdown table with two columns: Condition | ICD-10 Code."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You map symptoms to VA-claimable conditions and ICD-10 codes.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = response.choices[0].message.content or ""
        lines = content.strip().split("\n")

        data = []
        for line in lines:
            if not line.strip():
                continue
            if "Condition" in line and "ICD" in line:
                continue
            if set(line.strip()) <= {"|", "-", " "}:
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) == 2:
                data.append(parts)

        if not data:
            return pd.DataFrame({"Message": ["Could not parse response. Try different wording."]})

        return pd.DataFrame(data, columns=["Condition", "ICD-10 Code"])
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def summarize_evidence(text: str, user: str):
    error = require_client()
    if error:
        return error

    if not text:
        return "No text found in document."

    prompt = (
        "Summarize this medical evidence for a VA disability claim. "
        "Highlight symptoms, diagnoses, treatments, severity statements, and any service-connection clues.\n\n"
        f"{text}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You summarize medical documents for VA disability claims.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=800,
        )
        summary = response.choices[0].message.content

        cursor = conn.cursor()
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO claims (date, user, evidence_summary) VALUES (?, ?, ?)",
            (date, user, summary),
        )
        conn.commit()

        return summary
    except Exception as e:
        return f"Error summarizing evidence: {e}"

# --- Notes and claims ---
def save_note(user: str, content: str) -> str:
    if not content:
        return "Note content cannot be empty."
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO notes (timestamp, user, content) VALUES (?, ?, ?)",
        (timestamp, user, content),
    )
    conn.commit()
    return "Note saved."

def load_notes(user: str) -> str:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, content FROM notes WHERE user = ? ORDER BY timestamp DESC",
        (user,),
    )
    rows = cursor.fetchall()
    if not rows:
        return "No notes yet."
    parts = []
    for ts, content in rows:
        parts.append(f"**{ts}**\n{content}")
    return "\n\n---\n\n".join(parts)

def load_claims(user: str) -> pd.DataFrame:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, date, conditions, statement, evidence_summary FROM claims WHERE user = ? ORDER BY date DESC",
        (user,),
    )
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ID", "Date", "Type", "Content"])

    data = []
    for claim_id, date, conditions, statement, summary in rows:
        if statement:
            ctype = "Personal Statement"
            content = f"Conditions: {conditions}\n\n{statement[:300]}..."
        elif summary:
            ctype = "Evidence Summary"
            content = summary[:400] + "..."
        else:
            continue
        data.append([claim_id, date, ctype, content])

    return pd.DataFrame(data, columns=["ID", "Date", "Type", "Content"])

# --- Streamlit UI ---
def show_login():
    st.header("VA ClaimMate Login")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = ""

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Log in"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    with tab_register:
        new_user = st.text_input("New username", key="reg_user")
        new_pass = st.text_input("New password", type="password", key="reg_pass")
        if st.button("Create account"):
            msg = register_user(new_user, new_pass)
            if "Account created" in msg:
                st.success(msg)
            else:
                st.error(msg)

def show_main_app():
    user = st.session_state.user
    st.sidebar.write(f"Logged in as: {user}")
    if st.sidebar.button("Log out"):
        st.session_state.logged_in = False
        st.session_state.user = ""
        st.experimental_rerun()

    st.title("VA ClaimMate")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Statement Generator", "Evidence Summarizer", "Symptom Mapper", "My Notes", "Saved Claims"]
    )

    with tab1:
        st.subheader("Personal Statement Generator")
        name = st.text_input("Full Name")
        file_number = st.text_input("VA File Number")
        service_branch = st.selectbox(
            "Branch of Service",
            ["", "Army", "Navy", "Air Force", "Marine Corps", "Coast Guard", "Space Force"],
        )
        service_years = st.text_input("Years of Service", placeholder="e.g., 8")
        conditions = st.text_input(
            "Conditions you are claiming",
            placeholder="e.g., Tinnitus, PTSD, Knee pain",
        )
        symptom_description = st.text_area(
            "Symptoms and impact",
            placeholder="Describe how your symptoms affect work, daily life, sleep, relationships.",
            height=200,
        )

        if st.button("Generate statement"):
            result = generate_statement(
                service_branch,
                service_years,
                conditions,
                symptom_description,
                name,
                file_number,
                user,
            )
            st.write("Generated statement:")
            st.text_area("Statement", result, height=300)

        existing = st.text_area("Paste an existing statement to improve", height=200)
        if st.button("Rewrite statement"):
            improved = rewrite_statement(existing)
            st.write("Improved version:")
            st.text_area("Improved Statement", improved, height=300)

    with tab2:
        st.subheader("Evidence Summarizer")
        uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
        if uploaded_file is not None:
            if st.button("Summarize document"):
                raw_text = extract_text(uploaded_file)
                summary = summarize_evidence(raw_text, user)
                st.write("Summary:")
                st.text_area("AI Summary", summary, height=250)
                st.write("Extracted Text:")
                st.text_area("Document Text", raw_text, height=250)

    with tab3:
        st.subheader("Symptom to Condition Mapper")
        symptoms = st.text_area(
            "Symptoms",
            placeholder="e.g., ringing in ears, trouble sleeping, flashbacks, joint stiffness",
            height=150,
        )
        if st.button("Find conditions"):
            df = map_symptoms_to_conditions(symptoms)
            st.dataframe(df, use_container_width=True)

    with tab4:
        st.subheader("My Notes")
        note = st.text_area("New note", height=150)
        if st.button("Save note"):
            msg = save_note(user, note)
            if "saved" in msg:
                st.success(msg)
            else:
                st.error(msg)
        if st.button("Refresh notes"):
            notes_text = load_notes(user)
            st.write(notes_text)

    with tab5:
        st.subheader("Saved Claims")
        if st.button("Refresh saved claims"):
            df = load_claims(user)
            st.dataframe(df, use_container_width=True)

def main():
    setup_database()
    st.set_page_config(page_title="VA ClaimMate", layout="wide")

    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        show_login()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
