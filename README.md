# Clinical Text Simplification using Mistral-7B

This project fine-tunes Mistral-7B-Instruct-v0.2 to convert complex clinical and biomedical text into plain language that a patient can understand. The full pipeline covers dataset preparation, supervised fine-tuning with QLoRA, multi-metric evaluation, inference optimization, and a deployed Gradio demo.

Live demo: https://huggingface.co/spaces/prabhal/mistral-clinical-simplifier

Model: https://huggingface.co/prabhal/mistral-clinical-simplifier


## Motivation

Clinical notes and discharge summaries are written for clinicians, not patients. A sentence like "The patient presents with acute onset of chest pain radiating to the left arm, associated with dyspnea and diaphoresis, suggestive of a possible myocardial infarction" carries critical information that most patients cannot parse under stress. This project explores whether a fine-tuned language model can reliably bridge that gap.


## Dataset

The training data was sourced from the PubMed RCT 20k dataset, which contains structured sentences from biomedical research abstracts. The pipeline works as follows:

1. Sentences longer than 80 characters were extracted from the training split to filter out short, uninformative fragments.
2. A cleaning step removed annotation artifacts like "@", "-LSB-", and "-RSB-" tokens that appear in the raw dataset.
3. Sentence-level and paragraph-level inputs were constructed separately. Paragraphs were formed by joining three consecutive cleaned sentences, giving the model exposure to multi-sentence clinical context.
4. The combined input pool was used to generate a simplification dataset where each example has an instruction, an input (the clinical text), and an output (the simplified version).
5. A 90/10 train-validation split was applied before training.

Total training examples: approximately 400 (sentence and paragraph inputs combined).


## Model and Training

Base model: mistralai/Mistral-7B-Instruct-v0.2

Fine-tuning method: Supervised Fine-Tuning (SFT) using QLoRA, implemented with the Hugging Face TRL and PEFT libraries on a T4 GPU in Google Colab.

LoRA configuration:

1. Rank (r): 16
2. Alpha: 32
3. Target modules: q_proj, k_proj, v_proj, o_proj
4. Dropout: 0.05
5. Task type: Causal Language Modeling

Training configuration:

1. Epochs: 3
2. Learning rate: 2e-4 with cosine scheduler and 50 warmup steps
3. Batch size: 2 per device with 4 gradient accumulation steps (effective batch size 8)
4. Optimizer: paged_adamw_8bit
5. Precision: fp16
6. Max sequence length: 512 tokens

The prompt format used during training and inference follows the Alpaca-style instruction template with instruction, input, and response fields, applied through Mistral's chat template.


## Evaluation

Evaluation was run on 20 held-out samples comparing the fine-tuned model against the base Mistral-7B without any fine-tuning. Four types of metrics were used.

Readability (Flesch-Kincaid Grade Level):

| Text | Grade Level |
|---|---|
| Original clinical text | 15.58 |
| Base model output | 12.51 |
| Fine-tuned model output | 7.47 |

The fine-tuned model brings the reading level down to roughly middle-school level, which is the broadly recommended target for patient-facing health communication. The base model does reduce complexity somewhat, but the fine-tuned model achieves a much larger and more consistent drop.

Lexical Similarity (ROUGE scores against reference simplifications):

| Model | ROUGE-1 | ROUGE-L |
|---|---|---|
| Base model | 0.3773 | 0.2520 |
| Fine-tuned | 0.5274 | 0.3872 |

ROUGE-1 improved by roughly 40 percent and ROUGE-L by roughly 54 percent, indicating that the fine-tuned model produces outputs much closer in unigram and sequence overlap to the reference simplifications.

Semantic Similarity (BERTScore F1):

| Model | BERTScore F1 |
|---|---|
| Base model | 0.8878 |
| Fine-tuned | 0.9034 |

The improvement here is smaller in absolute terms, which is expected. BERTScore captures meaning-level alignment rather than surface overlap, and the base model is already a strong language model. The fine-tuned model nudges this further, showing that the simplifications are semantically closer to the references without introducing drift.

LLM-as-Judge (GPT evaluated on 1 to 10 scale):

| Dimension | Fine-tuned Model Score |
|---|---|
| Accuracy | 6.55 / 10 |
| Simplicity | 8.60 / 10 |
| Faithfulness | 6.45 / 10 |

Simplicity scores are high, confirming that the model reliably produces patient-friendly language. Accuracy and faithfulness sit in the mid-range, reflecting the inherent difficulty of preserving exact medical meaning while simplifying vocabulary. This points to a meaningful area for future improvement, particularly around grounding the model's outputs using clinical retrieval.


## Inference Optimization

After training, the LoRA adapter was merged back into the base model weights using the merge_and_unload method from PEFT. This eliminates the adapter overhead at inference time and produces a single standalone model checkpoint that can be served without any PEFT dependencies.

A latency comparison was run on a single T4 GPU between the adapter-based and merged configurations to confirm the speedup. Basic guardrails were also added to catch and flag outputs that attempt to provide a diagnosis, redirecting the user to treat the output as educational rather than medical advice.


## Repository Structure

1. sft_fine_tuning.ipynb: Full pipeline notebook covering environment setup, dataset construction, SFT training, evaluation, and inference optimization.
2. evaluation_results.csv: Per-sample evaluation outputs including clinical text, reference, baseline output, fine-tuned output, grade levels, and LLM judge scores for all 20 evaluation examples.
3. evaluation_summary.csv: Aggregated metric scores across all evaluation dimensions.


## Limitations

1. The training set is small. Around 400 examples is enough to shift the model's output style meaningfully, but not enough to guarantee factual precision across all clinical subdomains.
2. The reference simplifications used for ROUGE and BERTScore evaluation were generated by a teacher model rather than written by clinicians, which means the evaluation references themselves carry some noise.
3. Faithfulness scores from the LLM judge averaged 6.45 out of 10, indicating that hallucination and meaning drift remain real risks. This system should not be used in clinical settings without human review.
4. The Gradio demo runs on shared infrastructure and response latency will vary depending on load.


## Future Work

1. Expanding the training set with clinician-written simplifications from datasets like MedQuAD or discharge summary corpora would improve both faithfulness and accuracy.
2. Combining this fine-tuned model with a retrieval component (RAG over clinical guidelines) could reduce hallucination by grounding outputs in verified sources.
3. Running evaluation against Flesch Reading Ease in addition to Flesch-Kincaid grade would give a more complete picture of patient accessibility.
