# 📘 Romanized POS Tagger — English + Bengali (Roman Script)

A Machine Learning-based POS (Part-of-Speech) Tagger that labels each word in Romanized Bengali (written in English letters) with its grammatical role. Built using Python, scikit-learn, and Streamlit.

---

##  Features

* ✍️ Input: Bengali sentences in Roman script (e.g., "ami tomake bhalobashi")
* 🏷️ Output: POS tag for each word (e.g., PR\_PRP, N\_NN, V\_VM)
* 🧠 ML Model: Trained using LinearSVC
* 💻 Interface: Clean and interactive Streamlit web app

---

##  Who Can Use This Project?

* 📚 Students and researchers studying Natural Language Processing (NLP)
* 🧑‍💻 Beginners learning how to apply ML to linguistic data
* 🧪 Linguists exploring Romanized language tagging
* 🧑‍🏫 Teachers needing a demo project for ML or NLP
* 🗣️ Developers building multilingual language tools

---

## 📂 Project Structure

```bash
romanized-pos-tagger/
├── cleaned_data.txt        # Word-tag dataset (tab-separated)
├── train_model.py          # Trains the ML model and saves encoders
├── pos_tagger.py           # CLI-based POS tagging
├── pos_ui.py               # Streamlit web interface
├── model.pkl               # Trained LinearSVC model
├── word_enc.pkl            # Encoder for words
├── tag_enc.pkl             # Encoder for POS tags
├── requirements.txt        # All required libraries
└── README.md               # Project documentation
```

---

## Technologies Used

* Python 3.x
* scikit-learn
* joblib
* numpy & pandas
* Streamlit

---

##  How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model (if needed)

```bash
python train_model.py
```

### 3️⃣ Launch the web interface

```bash
streamlit run pos_ui.py
```

---

## Example

**Input:**

```
ami tomake bhalobashi
```

**Output:**

```
ami        → PR_PRP
tomake     → PR_PRP
bhalobashi → V_VM
```

---

##  Dataset Format

* Format: `word<TAB>tag`
* Example:

```
ami	PR_PRP
tomake	PR_PRP
bhalobashi	V_VM
```
---

## 💡 Notes

* `UNK` tag is used for unknown words

---

