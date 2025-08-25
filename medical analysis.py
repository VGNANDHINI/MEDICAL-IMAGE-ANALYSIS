import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# ------------------- Custom CSS for vibrant UI -------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #ff7eb3, #ff758c, #ff7e5f);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #fff;
}
h1 {
    text-align: center;
    font-size: 3em;
    text-shadow: 2px 2px 5px #000;
}
h2, h3 {
    color: #fff;
}
.stButton>button {
    background: linear-gradient(to right, #6a11cb, #2575fc);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    transition: transform 0.2s;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(to right, #2575fc, #6a11cb);
}
.stFileUploader {
    border: 2px dashed #fff;
    border-radius: 10px;
    padding: 15px;
    background: rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------------------------

# Set your API Key (Replace with your actual key)
GOOGLE_API_KEY = "AIzaSyBKKC8cWwEzLnLco2rQpA-JLRx45tO5eyE"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Ensure API Key is provided
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in GOOGLE_API_KEY")

# Initialize the Medical Agent
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Medical Analysis Query with Explainable AI Integration
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality and technical adequacy.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical/urgent findings.

### 4. Explainable AI (XAI)
- Explain why each finding leads to your diagnosis.
- Describe image features that influenced the AI decision.
- Present reasoning in a stepwise, transparent manner.

### 5. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

### 6. Research Context
- Provide 2-3 unique key references supporting the analysis.
- Ensure no repeated references in the output.
"""

# Function to analyze medical image
def analyze_medical_image(image_path):
    """Processes and analyzes a medical image using AI with XAI explanations."""
    
    # Open and resize image
    image = PILImage.open(image_path)
    width, height = image.size
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height))

    # Save resized image
    temp_path = "temp_resized_image.png"
    resized_image.save(temp_path)

    # Create AgnoImage object
    agno_image = AgnoImage(filepath=temp_path)

    # Run AI analysis
    try:
        response = medical_agent.run(query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
    finally:
        # Clean up temporary file
        os.remove(temp_path)

# Streamlit UI setup
st.set_page_config(page_title="Medical Image Analysis", layout="centered")
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown(
    """
    Welcome to the **Medical Image Analysis** tool! 📸
    Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and our AI-powered system will analyze it, providing detailed findings, Explainable AI insights, diagnosis, and research references.
    Let's get started!
    """
)

# Upload image section
st.sidebar.header("Upload Your Medical Image:")
uploaded_file = st.sidebar.file_uploader("Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "gif"])

# Button to trigger analysis
if uploaded_file is not None:
    # Display the uploaded image in Streamlit
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.sidebar.button("Analyze Image"):
        with st.spinner("🔍 Analyzing the image... Please wait."):
            # Save the uploaded image to a temporary file
            image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Run analysis on the uploaded image
            report = analyze_medical_image(image_path)
            
            # Display the report in a vibrant card
            st.markdown(f"""
            <div style="
                background: rgba(0,0,0,0.5); 
                padding: 20px; 
                border-radius: 15px; 
                margin-bottom: 20px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            ">
            <h3>📋 Analysis Report with Explainable AI</h3>
            <p>{report}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Clean up the saved image file
            os.remove(image_path)
else:
    st.warning("⚠️ Please upload a medical image to begin analysis.")
