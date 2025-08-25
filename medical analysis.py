import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# Set your API Key (Replace with your actual key)
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in GOOGLE_API_KEY")

# Initialize the Medical Agent
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

query = """YOUR_MEDICAL_ANALYSIS_QUERY_HERE"""

def analyze_medical_image(image_path):
    image = PILImage.open(image_path)
    width, height = image.size
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    temp_path = "temp_resized_image.png"
    resized_image.save(temp_path)
    agno_image = AgnoImage(filepath=temp_path)
    try:
        response = medical_agent.run(query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
    finally:
        os.remove(temp_path)

# Streamlit UI
st.set_page_config(page_title="Medical Image Analysis", layout="wide")

# --- Inject custom CSS for gradients ---
# --- CUSTOM CSS FOR VIBRANT FRONTEND ---
st.markdown("""
<style>
/* ---------- FULL PAGE BACKGROUND ---------- */
.stApp {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #a18cd1 100%);
    font-family: 'Arial', sans-serif;
}

/* ---------- SIDEBAR BACKGROUND ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffecd2 0%, #fcb69f 100%);
    padding: 20px;
    border-radius: 12px;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: linear-gradient(to right, #f6d365, #fda085);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 24px;
    transition: 0.3s;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}
.stButton>button:hover {
    background: linear-gradient(to right, #f093fb, #f5576c);
    transform: scale(1.05);
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    color: #ff6f61;
    text-shadow: 2px 2px 5px #000000;
}

/* ---------- FILE UPLOADER ---------- */
.css-1y4p8pa {  /* may vary with Streamlit version */
    border: 2px dashed #fbc2eb;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.2);
    padding: 10px;
}

/* ---------- REPORT / OUTPUT CARDS ---------- */
.report-card {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    color: #333333;
}

/* ---------- FOOTER / SMALL TEXT ---------- */
footer {
    color: #333333;
    font-size: 12px;
    text-align: center;
    padding-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title & description
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown("""
Welcome to the **Medical Image Analysis** tool! 📸  
Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and our AI system will provide detailed findings, Explainable AI insights, and references.  
""")

# Sidebar for uploading
st.sidebar.header("Upload Your Medical Image:")
uploaded_file = st.sidebar.file_uploader("Choose a medical image", type=["jpg","jpeg","png","bmp","gif"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    if st.sidebar.button("Analyze Image"):
        with st.spinner("🔍 Analyzing image..."):
            image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            report = analyze_medical_image(image_path)
            st.subheader("📋 Analysis Report with Explainable AI")
            st.markdown(report, unsafe_allow_html=True)
            os.remove(image_path)
else:
    st.warning("⚠️ Please upload a medical image to begin analysis.")
