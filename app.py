import streamlit as st
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Clinical Text Simplifier", page_icon="🏥")

st.title("🏥 Clinical Text Simplifier")
st.caption("Fine-tuned Mistral-7B · Converts clinical language into patient-friendly explanations")

MODEL_ID = "prabhal/mistral-clinical-simplifier"

# Read token from Streamlit secrets
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = None
    st.warning("HF_TOKEN not found in secrets.")

@st.cache_resource
def get_client(token):
    return InferenceClient(token=token)

client = get_client(HF_TOKEN)

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
                st.error(f"Inference failed: {type(e).__name__}: {e}")
                st.info(
                    "The free HF Serverless Inference API may not support this model. "
                    "Try visiting the model page to wake it up: "
                    "https://huggingface.co/prabhal/mistral-clinical-simplifier"
                )

st.caption("⚕️ For educational purposes only. Not medical advice.")
