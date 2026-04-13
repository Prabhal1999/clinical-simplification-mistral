import streamlit as st
import requests
import os

st.set_page_config(page_title="Clinical Text Simplifier")

st.title("Clinical Text Simplifier")
st.caption("Fine-tuned Mistral-7B - Converts clinical language into patient-friendly explanations")

try:
    MODAL_URL = st.secrets["MODAL_URL"]
except Exception:
    MODAL_URL = None
    st.warning("MODAL_URL not found in secrets.")

input_text = st.text_area("Clinical Text", placeholder="Paste clinical or medical text here...", height=180)

if st.button("Simplify"):
    if not input_text.strip():
        st.warning("Please enter some clinical text first.")
    elif not MODAL_URL:
        st.error("MODAL_URL is not configured.")
    else:
        with st.spinner("Simplifying... (first request may take up to 60s to cold start)"):
            try:
                response = requests.post(
                    MODAL_URL,
                    json={"text": input_text.strip()},
                    timeout=180,
                )
                if response.status_code == 200:
                    output = response.json().get("output", "")
                    st.text_area("Simplified Output", value=output, height=180)
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

st.caption("For educational purposes only. Not medical advice.")
