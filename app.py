import streamlit as st
import json
import random
from transformers import pipeline

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ----------------------------
# LOAD MODEL (LOW RAM SAFE)
# ----------------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model=MODEL_NAME, device=-1)

llm = load_model()

# ----------------------------
# LOAD DATA
# ----------------------------
with open("knowledge_chunks.json", "r", encoding="utf-8") as f:
    KNOWLEDGE = json.load(f)

with open("scenarios.json", "r", encoding="utf-8") as f:
    SCENARIOS = json.load(f)

# ----------------------------
# UTILS
# ----------------------------
def retrieve_knowledge():
    return " ".join([k["text"] for k in random.sample(KNOWLEDGE, 2)])

def generate(prompt, max_tokens=250):
    out = llm(prompt, max_new_tokens=max_tokens, temperature=0.6)
    return out[0]["generated_text"]

# ----------------------------
# SESSION STATE
# ----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = None

# ----------------------------
# UI
# ----------------------------
st.title("🌱 Sustainable Digitalization – AI Tutor")

st.markdown("""
This interactive chatbot helps you test knowledge and reason through
real-world sustainability scenarios.
""")

choice = st.radio(
    "How would you like to proceed?",
    ["Knowledge Check", "Scenario-Based Activity"]
)

# ----------------------------
# KNOWLEDGE CHECK MODE
# ----------------------------
if choice == "Knowledge Check":
    question = st.text_input("Enter a sustainability-related question:")

    answer = st.text_area("Your answer:")

    if st.button("Submit Answer"):
        context = retrieve_knowledge()

        prompt = f"""
SYSTEM:
You are an academic tutor for Sustainable Digitalization.

CONTEXT:
{context}

QUESTION:
{question}

STUDENT ANSWER:
{answer}

TASK:
1. Say if the answer is correct.
2. Correct it if needed.
3. Ask one follow-up question.
"""
        response = generate(prompt)
        st.markdown("### AI Feedback")
        st.write(response)

# ----------------------------
# SCENARIO MODE
# ----------------------------
if choice == "Scenario-Based Activity":
    scenario = random.choice(SCENARIOS)

    st.markdown("### Business Scenario")
    st.write(scenario["scenario"])

    student_solution = st.text_area("Your proposed solution:")

    if st.button("Evaluate Solution"):
        prompt = f"""
SYSTEM:
You are an expert tutor in sustainable digitalization.

SCENARIO:
{scenario["scenario"]}

STUDENT RESPONSE:
{student_solution}

TASK:
1. Identify one sustainability benefit
2. Identify one trade-off or risk
3. Ask ONE probing follow-up question
"""
        feedback = generate(prompt)
        st.markdown("### AI Feedback")
        st.write(feedback)

        reflection = st.text_area("Reflection (optional):")

        if reflection:
            summary_prompt = f"""
SYSTEM:
You are an academic evaluator.

STUDENT REFLECTION:
{reflection}

TASK:
Summarize the key learning in 3–4 lines.
"""
            summary = generate(summary_prompt)
            st.markdown("### Learning Summary")
            st.write(summary)

