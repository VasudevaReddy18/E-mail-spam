import streamlit as st
from models.model_trainer import ModelTrainer

st.title("📧 Email Spam Classifier (Streamlit)")
st.write("Enter the email content below to check if it's spam or not:")

# Load the trained model (cache for performance)
@st.cache_resource
def load_trainer():
    trainer = ModelTrainer()
    trainer.load_model()
    return trainer

trainer = load_trainer()

email_content = st.text_area("Email Content", height=200)

if st.button("Classify"):
    if not email_content.strip():
        st.warning("Please enter some email content.")
    else:
        result = trainer.test_model([email_content])[0]
        st.markdown(f"**Prediction:** {'🛑 SPAM' if result['is_spam'] else '✅ HAM'}")
        st.markdown(f"**Spam Probability:** {result['spam_probability']*100:.2f}%")
        st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%") 