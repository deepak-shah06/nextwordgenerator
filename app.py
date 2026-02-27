import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ------------------------------------------------------
# Utility: Create index_to_word dictionary
# ------------------------------------------------------
def create_index_to_word(tokenizer):
    index_to_word = {}
    for word, index in tokenizer.word_index.items():
        index_to_word[index] = word
    return index_to_word


# ------------------------------------------------------
# Predictor Function
# ------------------------------------------------------
def predictor(model, tokenizer, text, max_len, index_to_word):
    text = text.lower()

    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')

    pred = model.predict(seq, verbose=0)
    pred_index = np.argmax(pred)

    if pred_index == 0:
        return ""

    return index_to_word.get(pred_index, "")


# ------------------------------------------------------
# Generate Text Function
# ------------------------------------------------------
def generate_text(model, tokenizer, seed_text, max_len, n_words, index_to_word):
    generated_text = seed_text
    for _ in range(n_words):
        next_word = predictor(model, tokenizer, generated_text, max_len, index_to_word)
        if not next_word:
            break
        generated_text += " " + next_word
    return generated_text


# ------------------------------------------------------
# Load Assets Once (Cached)
# ------------------------------------------------------
@st.cache_resource
def load_assets():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    index_to_word = create_index_to_word(tokenizer)

    return model, tokenizer, max_len, index_to_word


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------
st.title("✨ Quote Generator (LSTM Model)")

st.write("Enter a starting phrase and choose how many words you want to generate.")

# Load model + assets
model, tokenizer, max_len, index_to_word = load_assets()

# User Inputs
seed_text = st.text_input("Enter seed text:", "The best way to")
n_words = st.slider("Number of words to generate:", min_value=1, max_value=50, value=10)

# Generate Button
if st.button("Generate Quote"):
    output = generate_text(model, tokenizer, seed_text, max_len, n_words, index_to_word)
    st.subheader("Generated Text:")
    st.write(output)