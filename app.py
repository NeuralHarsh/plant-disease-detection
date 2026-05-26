import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load trained model
model = load_model("plant_disease_model.h5")

# Class names
CLASS_NAMES = (
    'Tomato-Bacterial_spot',
    'Potato-Early_blight',
    'Corn-Common_rust'
)

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection")
st.sidebar.markdown("""
### Deep Learning Final Year Project

This system uses a CNN (Convolutional Neural Network)
to detect diseases in plant leaves.

### Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
""")

st.sidebar.success("Model Loaded Successfully")

# Main Title
st.title("🌿 Plant Disease Detection using CNN")
st.markdown("---")

# About section
with st.expander("📖 About This Project"):
    st.write("""
    This application predicts plant diseases using
    Deep Learning techniques.

    Upload an image of a plant leaf and the model
    will identify the disease along with confidence score.
    """)

# File uploader
plant_image = st.file_uploader(
    "📤 Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# Predict button
if st.button("🔍 Predict Disease"):

    if plant_image is not None:

        # Convert uploaded file to OpenCV image
        file_bytes = np.asarray(
            bytearray(plant_image.read()),
            dtype=np.uint8
        )

        opencv_image = cv2.imdecode(file_bytes, 1)

        # Create two columns
        col1, col2 = st.columns(2)

        # Display image
        with col1:
            st.image(
                opencv_image,
                channels="BGR",
                caption="Uploaded Leaf Image",
                use_container_width=True
            )

        # Resize image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Normalize image
        opencv_image = opencv_image / 255.0

        # Convert image to 4D
        opencv_image = opencv_image.reshape(1, 256, 256, 3)

        # Prediction
        Y_pred = model.predict(opencv_image)

        prediction = np.argmax(Y_pred)

        result = CLASS_NAMES[prediction]

        confidence = np.max(Y_pred) * 100

        # Split plant and disease
        plant, disease = result.split('-')

        # Disease information
        disease_info = {
            "Bacterial_spot":
                "Bacterial infection causing dark spots on leaves.",

            "Early_blight":
                "Fungal disease causing brown circular spots.",

            "Common_rust":
                "Fungal disease creating rust-colored pustules."
        }

        # Treatment suggestions
        treatment = {
            "Bacterial_spot":
                "Use copper-based bactericides and remove infected leaves.",

            "Early_blight":
                "Apply fungicides and avoid overwatering.",

            "Common_rust":
                "Use resistant crop varieties and fungicides."
        }

        # Prediction results
        with col2:

            st.subheader("🩺 Prediction Result")

            st.success(f"🌿 Plant: {plant}")

            st.error(f"🦠 Disease: {disease}")

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

            st.info(
                f"📖 Disease Info:\n\n{disease_info.get(disease)}"
            )

            st.warning(
                f"💊 Treatment:\n\n{treatment.get(disease)}"
            )

    else:
        st.warning("⚠️ Please upload an image first.")

# Footer
st.markdown("---")
st.caption("Final Year Project by Kite Nitesh Ramkrushna | Plant Disease Detection using Deep Learning")
