import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st
import sqlite3
import hashlib
#-------------------------------
## --- SESSION STATE INIT ---
#---------------------------------
if 'login' not in st.session_state:
    st.session_state['login'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = ""
if 'guest' not in st.session_state:
    st.session_state['guest'] = False

#-------------------------------
## --- new login setup ---
#---------------------------------

if not st.session_state['login'] and not st.session_state['guest']:
    st.title("Welcome to Medical Image Analyzer")
    choice = st.radio("Login / Sign Up / Continue as Guest", ["Login", "Sign Up", "Guest"])
    
    if choice == "Sign Up":
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            c.execute("SELECT * FROM users WHERE username = ?", (new_user,))
            if c.fetchone():
                st.warning("Username already exists! Choose a different one.")
            else:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, hash_password(new_pass)))
                conn.commit()
                st.success("Account created! Please log in.")
    
    elif choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            c.execute("SELECT password FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result and verify_password(result[0], password):
                st.session_state['login'] = True
                st.session_state['user'] = username
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password")
    
    elif choice == "Guest":
        if st.button("Continue as Guest"):
            st.session_state['guest'] = True
            st.experimental_rerun()

    # --- EXISTING IMAGE UPLOAD & ANALYSIS CODE ---
    st.sidebar.header("Upload Your Medical Image:")
    uploaded_file = st.sidebar.file_uploader("Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "gif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.sidebar.button("Analyze Image"):
            with st.spinner("Analyzing the image... 🔍"):
                analysis_text, display_image = analyze_medical_image(uploaded_file)
            
            if display_image:
                st.image(display_image, caption="Resized Image", use_container_width=True)
            
            st.subheader("📝 AI Analysis Report")
            st.markdown(analysis_text)








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
query_template = """
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
def analyze_medical_image(image_file):
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

        # Run AI analysis
        response = medical_agent.run(query_template, images=[agno_image])
        content = response.content if hasattr(response, "content") else str(response)

        # Remove repeated sections if any (safety check)
        if "### 5. Research Context" in content:
            main_report, research_context = content.split("### 5. Research Context", 1)
            content = main_report.strip() + "\n### 5. Research Context" + research_context.strip().split("### 5. Research Context")[-1]

        return content, resized_image

    except Exception as e:
        return f"Analysis error: {e}", None

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# -------------------------------







# 5️⃣ Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Medical Image Analysis", layout="centered")
st.title("🩺 Medical Image Analysis Tool 🔬")

# Sidebar menu for Login / Sign Up---------------------------------------------------------------------
menu = ["Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Sign Up":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    if st.button("Sign Up"):
        success, msg = sign_up(new_user, new_password)
        if success:
            st.success("Account created successfully!")  # <-- hard-coded
        else:
            st.error(str(msg))  # ensure msg is string

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        success, msg = login(username, password)
        if success:
            st.success("Login successful!")  # <-- hard-coded
            st.session_state['login'] = True
            st.session_state['user'] = str(username)
        else:
            st.error(str(msg))  # ensure msg is string

#--------------------------------------------------------------------------
if st.session_state.get('login'):
    # All your existing app code goes here
    # Image upload, AI analysis, annotations, report generation, etc.
    st.write(f"Welcome, {st.session_state['user']}! You can now use the app.")
# LOGOUT BUTTON--------------------------------------------------------------
    if st.sidebar.button("Log Out"):
        st.session_state['login'] = False
        st.session_state['user'] = ""
        st.success("You have been logged out successfully.")
#--------------------------------------------------------------------------------

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
