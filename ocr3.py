import streamlit as st
import sqlite3
import bcrypt
import pytesseract
from PIL import Image
import cv2
import numpy as np
import spacy
from gtts import gTTS
from io import BytesIO

# Initialize SQLite database
conn = sqlite3.connect('users.db')
c = conn.cursor()

# Create users table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def signup(username, password):
    # Check if user already exists
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        return False
    else:
        # Hash the password and store the user
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True

def login(username, password):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    if user and verify_password(password, user[1]):
        return True
    else:
        return False

def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_image(img):
    # Convert image to grayscale if it's not already
    if len(np.array(img).shape) == 3:
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    else:
        gray_img = np.array(img)
    # Apply thresholding
    _, thresh_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh_img

def detect_and_recognize_text(img, lang='eng', is_handwritten=False):
    # Preprocess the image for text recognition
    preprocessed_img = preprocess_image(img)

    # Use Tesseract to detect text boxes on the preprocessed image
    boxes = pytesseract.image_to_boxes(preprocessed_img, lang=lang)

    # Draw bounding boxes on a copy of the original image (for display)
    display_img = np.array(img).copy()
    if len(display_img.shape) == 2:  # Grayscale image
        h, w = display_img.shape
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for drawing boxes
    else:  # Color image
        h, w, _ = display_img.shape

    for b in boxes.splitlines():
        b = b.split(' ')
        display_img = cv2.rectangle(display_img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # Use Tesseract to recognize text from the preprocessed image
    if is_handwritten:
        custom_config = f'--oem 1 --psm 6 -l {lang}'
    else:
        custom_config = f'--oem 3 --psm 6 -l {lang}'
    detected_text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

    return detected_text, display_img

def analyze_text(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Extract tokens, part-of-speech tags, and named entities
    tokens = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return tokens, pos_tags, entities
# Function to generate audio from text
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Streamlit UI for Authentication
st.title("OCRius with Authentication")

auth_status = st.session_state.get('auth_status', None)
if auth_status == "logged_in":
    st.success(f"Welcome {st.session_state.username}!")

    # OCR functionality
    language_options = {
        'English': 'eng',
        'French': 'fra',
        'German': 'deu',
        'Spanish': 'spa',
        'Telugu': 'tel',
        'Hindi': 'hin',
        # Add more languages and their corresponding Tesseract language codes here
    }
    selected_language = st.selectbox("Select Language", list(language_options.keys()))

    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    is_handwritten = st.checkbox("Is the text handwritten?")
    if image_file is not None:
        img = load_image(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                detected_text, annotated_img = detect_and_recognize_text(img, lang=language_options[selected_language], is_handwritten=is_handwritten)
                st.success("Text extracted successfully!")

            st.subheader("Extracted Text")
            st.write(detected_text)
            st.subheader("Annotated Image")
            st.image(annotated_img, use_column_width=True)
            audio_buffer = text_to_audio(detected_text)
            st.audio(audio_buffer, format='audio/mp3')


            tokens, pos_tags, entities = analyze_text(detected_text)
            st.subheader("NLP Analysis")
            st.write("Tokens:", tokens)
            st.write("Part-of-Speech Tags:", pos_tags)
            st.write("Named Entities:", entities)

elif auth_status == "login_failed":
    st.error("Login failed. Please check your username and password.")
    auth_status = None
elif auth_status == "signup_failed":
    st.error("Signup failed. Username already exists.")
    auth_status = None
# Login/Signup form
if auth_status is None or auth_status == "logged_out":
    form_type = st.radio("Choose form type:", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if form_type == "Login":
        if st.button("Login"):
            if login(username, password):
                st.session_state.auth_status = "logged_in"
                st.session_state.username = username
                st.rerun()
            else:
                st.session_state.auth_status = "login_failed"
                st.rerun()
    else:  # Signup
        if st.button("Signup"):
            if signup(username, password):
                st.session_state.auth_status = "logged_in"
                st.session_state.username = username
                st.rerun()
            else:
                st.session_state.auth_status = "signup_failed"
                st.rerun()

# Logout button
if auth_status == "logged_in":
    if st.button("Logout"):
        st.session_state.auth_status = "logged_out"
        del st.session_state.username
        st.rerun()
