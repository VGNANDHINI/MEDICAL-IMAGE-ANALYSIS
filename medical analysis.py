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
st.markdown("""
<style>
/* Main background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #d0f0fd, #d0fdd8);
}

/* Sidebar gradient */
[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #7fffd4, #2f2f2f);
}

/* Output/report section */
.css-1d391kg { 
    background: linear-gradient(to bottom, #003366, #6699ff);
    padding: 1rem;
    border-radius: 10px;
}

/* Button style */
.stButton>button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 8px;
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
