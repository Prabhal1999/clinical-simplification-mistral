import streamlit as st
import requests

st.set_page_config(page_title="Clinical Text Simplifier")

st.title("Clinical Text Simplifier")
st.caption("Fine-tuned Mistral-7B: Converts clinical language into patient-friendly explanations")

MODEL_ID = "prabhal/mistral-clinical-simplifier"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}/v1/completions"

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None
    st.warning("HF_TOKEN not found in secrets.")

input_text = st.text_area("Clinical Text", placeholder="Paste clinical or medical text here...", height=180)

if st.button("Simplify"):
    if not input_text.strip():
        st.warning("Please enter some clinical text first.")
    else:
        prompt = (
            "### Instruction:\nSimplify the following clinical text for a patient.\n\n"
            f"### Input:\n{input_text.strip()}\n\n### Response:\n"
        )
        with st.spinner("Simplifying... (may take up to 30s on first run)"):
            try:
                headers = {
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": MODEL_ID,
                    "prompt": prompt,
                    "max_tokens": 250,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "stop": ["###"]
                }

                response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
                st.caption(f"API status: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()
                    output = result["choices"][0]["text"].strip()
                    st.text_area("Simplified Output", value=output, height=180)

                elif response.status_code == 503:
                    wait = response.json().get("estimated_time", "unknown")
                    st.warning(f"Model is loading. Estimated wait: {wait}s. Please try again shortly.")

                else:
                    st.error(f"HF API error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")

st.caption("⚕️ For educational purposes only. Not medical advice.")
