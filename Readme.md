# 🩺  Medical Topic Discovery

This project demonstrates a practical application of **Unsupervised Machine Learning** to discover underlying themes within a collection of medical abstracts or notes. Using **Latent Dirichlet Allocation (LDA)**, the system automatically identifies topics (like "Oncology" or "Clinical Trials") without requiring any pre-labeled data.

---

## ✨ Project Highlights

* **Core Task:** Topic Modeling (Unsupervised Learning).
* **Vectorization:** Uses **Count Vectorization** to transform text into numerical features.
* **Modeling:** Employs **Latent Dirichlet Allocation (LDA)** to discover hidden topic structures.
* **Output:** Generates a list of the top representative words for each discovered topic, allowing for manual interpretation and labeling.

---

## 📁 Project Structure

* `medical-topic-discovery/`
    * `├── .gitignore`
    * `├── README.md`
    * `├── requirements.txt`
    * `├── data/`
        * `└── medical_corpus.txt` (Input text, one document per line)
    * `└── src/`
        * `└── model_trainer.py` (Main script for data prep, vectorization, and LDA modeling)


---

## 🚀 Getting Started

1.  **Clone** and install dependencies:
    ```bash
    git clone [https://github.com/yourusername/medical-topic-discovery.git](https://github.com/yourusername/medical-topic-discovery.git)
    cd medical-topic-discovery
    pip install -r requirements.txt
    ```

2.  **Prepare Data**: Ensure your medical documents are in `data/medical_corpus.txt`, with **one document per line**.

3.  **Run**:
    ```bash
    python src/model_trainer.py
    ```

---

## 🔑 Interpreting Results

The script prints the top words for each found topic. You interpret these words to manually assign a label.

| Example Topic | Top Words | Assigned Label |
| :---: | :--- | :--- |
| **Topic 1** | drug, trial, dose, patient | **Pharmacology** |
| **Topic 2** | heart, artery, surgery, blood | **Cardiology** |