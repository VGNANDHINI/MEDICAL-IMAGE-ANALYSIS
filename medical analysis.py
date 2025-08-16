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
#import database
#---------------------------------
# Connect to database (or create)
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if not exists
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()
# -------------------------------
# password
# -------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)
# -------------------------------
# sign up and login
# -------------------------------
def sign_up(username, password):
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                  (username, hash_password(password)))
        conn.commit()
        return True, "User registered successfully!"
    except sqlite3.IntegrityError:
        return False, "Username already exists."

def login(username, password):
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    if result:
        if verify_password(result[0], password):
            return True, "Login successful!"
        else:
            return False, "Incorrect password."
    else:
        return False, "User does not exist."









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
        st.success(msg) if success else st.error(msg)

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        success, msg = login(username, password)
        if success:
            st.success(str(msg))
            st.session_state['login'] = True
            st.session_state['user'] = username
        else:
            st.error(str(msg))

#--------------------------------------------------------------------------
if st.session_state.get('login'):
    # All your existing app code goes here
    # Image upload, AI analysis, annotations, report generation, etc.
    st.write(f"Welcome, {st.session_state['user']}! You can now use the app.")

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
