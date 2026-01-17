import streamlit as st
import backend
from streamlit_mermaid import st_mermaid

st.set_page_config(page_title="FlowLearn AI", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .reportview-container { background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ FlowLearn: Chart Your Path (Multi Agent Orchestration)", text_alignment='center')
st.subheader("Agent A: llama-3.1-8b (Acts as Professor 🎓)  |  Agent B: llama-3.1-8b (Acts as AI Engineer 💻)  |  Boss: llama-3.3-70b (Final Architect 🏁)")
st.subheader("Vision model used: meta-llama/llama-4-scout-17b-16e-instruct")
st.subheader("Audio model used: whisper-large-v3")
st.caption("Upload notes, speak, or type. We build the flowchart AND find the links.", text_alignment="center")

# Sidebar
with st.sidebar:
    st.header("🔑 Setup")
    api_key = st.text_input("Groq API Key", type="password")
    st.info("Agent A: Llama-3.1-8b\nAgent B: Llama-3.1-8b\nBoss: Llama-3.3-70b")
    st.markdown("[Get Key](https://console.groq.com/keys)")

# Inputs
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    txt_input = st.text_area("Topic / Goal", placeholder="e.g. Learn React Native from scratch...", height=150)
with col2:
    audio_input = st.audio_input("Voice Note")
with col3:
    img_input = st.file_uploader("Handwritten Plan", type=["jpg", "png", "jpeg"])

if st.button("🚀 Generate Roadmap"):
    if not api_key:
        st.error("Please provide an API Key first.")
    else:
        status_box = st.status("🧠 Processing...", expanded=True)
        
        try:
            # --- STEP 1: FLOWCHART ---
            status_box.write("1. Detecting Inputs & Consulting Agents (Gemma & Llama)...")
            mermaid_code, context = backend.generate_flowchart(api_key, txt_input, audio_input, img_input)
            
            if not mermaid_code:
                status_box.update(label="Error: No input detected", state="error")
                st.stop()
                
            status_box.write("2. Big Boss Architect is synthesizing the flowchart...")
            
            # Show Flowchart immediately
            st.subheader("1. The Blueprint")
            
            # Use full width container
            with st.container():
                st_mermaid(mermaid_code, height="800px")
            
            with st.expander("View Raw Mermaid Code"):
                st.code(mermaid_code)

            # --- STEP 2: RESOURCES ---
            status_box.write("3. Scouting the web for tutorials & videos...")
            resources = backend.find_learning_resources(api_key, mermaid_code, context)
            
            status_box.update(label="Complete!", state="complete", expanded=False)
            
            st.divider()
            st.subheader("2. The Toolkit (Verified Sources)")
            st.markdown(resources)

        except Exception as e:
            status_box.update(label="System Error", state="error")
            st.error(f"An error occurred: {str(e)}")