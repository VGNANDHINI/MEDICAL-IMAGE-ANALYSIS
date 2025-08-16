import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# -------------------------------
# 1️⃣ Set API Key
# -------------------------------
GOOGLE_API_KEY = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ Please set your Google API Key in GOOGLE_API_KEY")

# -------------------------------
# 2️⃣ Initialize the Medical Agent
# -------------------------------
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# -------------------------------
# 3️⃣ AI Query Template
# -------------------------------
query= """
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

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

### 5. Research Context
- Use DuckDuckGo search to find recent medical literature.
- Search for standard treatment protocols.
- Provide 2-3 key references supporting the analysis.

Also, provide explainable AI insights:
- Highlight image regions that contributed most to your analysis
- Provide confidence levels for each finding
- Give reasoning behind each diagnosis in simple terms
"""

# -------------------------------
# 4️⃣ Function to Analyze Image
# -------------------------------
def analyze_medical_image(image_path,retries=3,delay=5):
    """
    Processes and analyzes a medical image using AI.
    Returns the AI response text and a resized image for display.
    """
    temp_path = None
    try:
        # Open and resize image
        image = PILImage.open(image_file)
        width, height = image.size
        aspect_ratio = width / height
        new_width = 500
        new_height = int(new_width / aspect_ratio)
        resized_image = image.resize((new_width, new_height))

        # Save resized image temporarily for agno
        temp_path = "temp_resized_image.png"
        resized_image.save(temp_path)

        # Create agno image object
        agno_image = AgnoImage(filepath=temp_path)

     # Add explainable AI prompt
        xai_query = query + """
        Also, provide explainable AI insights:
        - Highlight regions contributing most to analysis
        - Confidence levels for each finding
        - Reasoning behind each diagnosis in simple terms
        """

        # Retry loop for 429 errors
        for attempt in range(retries):
            try:
                response = medical_agent.run(xai_query, images=[agno_image])
                return response.content, resized_image
            except Exception as e:
                if "429" in str(e):
                    st.warning(f"API limit reached. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return f"Analysis error: {e}", resized_image

        return "Analysis failed after multiple attempts due to API limits.", resized_image

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------------------
# 5️⃣ Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Medical Image Analysis", layout="centered")
st.title("🩺 Medical Image Analysis Tool 🔬")
st.markdown("""
Welcome to the Medical Image Analysis tool! 📸  
Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and our AI-powered system will analyze it, providing detailed findings, diagnosis, and research insights.
""")

# Sidebar upload
st.sidebar.header("Upload Your Medical Image:")
uploaded_file = st.sidebar.file_uploader("Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "gif"])

# Analyze button
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.sidebar.button("Analyze Image"):
        with st.spinner("Analyzing the image... 🔍"):
            analysis_text, display_image = analyze_medical_image(uploaded_file)
        
        if display_image:
            st.image(display_image, caption="Resized Image", use_container_width=True)
        
        st.subheader("📝 AI Analysis Report")
        st.markdown(analysis_text)
