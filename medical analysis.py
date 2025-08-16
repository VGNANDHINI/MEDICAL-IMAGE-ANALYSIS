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
