import streamlit as st
from huggingface_hub import InferenceClient
import os

st.set_page_config(page_title="Clinical Text Simplifier")

st.title("Clinical Text Simplifier")
st.caption("Fine-tuned Mistral-7B: Converts clinical language into patient-friendly explanations")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "prabhal/mistral-clinical-simplifier"

@st.cache_resource
def get_client():
    return InferenceClient(token=HF_TOKEN if HF_TOKEN else None)

client = get_client()

input_text = st.text_area("Clinical Text", placeholder="Paste clinical or medical text here...", height=180)

if st.button("Simplify"):
    if not input_text.strip():
        st.warning("Please enter some clinical text first.")
    else:
        prompt = (
            "### Instruction:\nSimplify the following clinical text for a patient.\n\n"
            f"### Input:\n{input_text.strip()}\n\n### Response:\n"
        )
        with st.spinner("Simplifying..."):
            try:
                response = client.text_generation(
                    prompt,
                    model=MODEL_ID,
                    max_new_tokens=250,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    stop_sequences=["###"],
                )
                st.text_area("Simplified Output", value=response.strip(), height=180)
            except Exception as e:
                st.error(f"Error: {e}")

st.caption("⚕️ For educational purposes only. Not medical advice.")
