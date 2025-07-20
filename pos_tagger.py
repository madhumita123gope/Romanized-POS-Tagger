# pos_tagger.py

import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("model.pkl")
word_enc = joblib.load("word_enc.pkl")
tag_enc = joblib.load("tag_enc.pkl")

def predict_pos(sentence):
    words = sentence.strip().split()
    output = []

    for word in words:
        try:
            encoded = word_enc.transform([word])
            pred = model.predict(np.array(encoded).reshape(-1, 1))
            tag = tag_enc.inverse_transform(pred)[0]
        except ValueError:
            tag = "UNK"  # Unknown word
        output.append((word, tag))
    
    return output

if __name__ == "__main__":
    print("🔠 Romanized POS Tagger (Type 'exit' to quit)\n")

    while True:
        sentence = input("Enter a sentence: ").strip()
        if sentence.lower() in ["exit", "quit"]:
            print("👋 Bye!")
            break

        result = predict_pos(sentence)
        print("\nPOS Tags:")
        for word, tag in result:
            print(f"{word:10} → {tag}")
        print("-" * 30)
