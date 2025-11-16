import os
import streamlit as st
from openai import OpenAI
import pandas as pd

# --- Configuration ---
API_KEY = os.environ.get("")
client = OpenAI(api_key=API_KEY)

# --- Symptom Mapping Function ---
def map_symptoms(symptoms):
    if not symptoms:
        return pd.DataFrame({"Message": ["Please enter symptoms first."]})

    prompt = (
        f"A U.S. veteran lists these symptoms for a VA disability claim: '{symptoms}'. "
        f"List the most likely VA-claimable medical conditions with their ICD-10 codes. "
        f"Respond as a markdown table with two columns: Condition | ICD-10 Code."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You map symptoms to VA-claimable conditions and ICD-10 codes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=400
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
            return pd.DataFrame({"Message": ["Could not parse response. Try different symptoms."]})

        return pd.DataFrame(data, columns=["Condition", "ICD-10 Code"])

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


# --- Streamlit UI ---
st.set_page_config(page_title="VA ClaimMate MVP1", layout="centered")

st.title("VA ClaimMate MVP1")
st.subheader("Symptom to Condition Mapper")

st.write("Enter your symptoms below. Example:")
st.code("ringing in ears, trouble sleeping, headaches, back pain")

symptoms = st.text_area("Symptoms", height=150)

if st.button("Find Potential Conditions"):
    df = map_symptoms(symptoms)
    st.write("### Results")
    st.dataframe(df)
