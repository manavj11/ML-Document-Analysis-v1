# 🩺  Doc-Analysis-v1

This project demonstrates a practical application of **Unsupervised Machine Learning** to discover underlying themes within a collection of medical abstracts or notes. Using **Latent Dirichlet Allocation (LDA)**, the system automatically identifies topics (like "Oncology" or "Treatment") without requiring any pre-labeled data.

---

## ✨ Project Highlights

* **Core Task:** Topic Modeling (Unsupervised Learning).
* **Vectorization:** Uses **Count Vectorization** to transform text into numerical features.
* **Modeling:** Employs **Latent Dirichlet Allocation (LDA)** to discover hidden topic structures.
* **Output:** Generates a list of the top representative words for each discovered topic, allowing for manual interpretation and labeling.

---

## 📁 Project Structure

* `Doc-Analysis-v1/`
    * `├── .gitignore`
    * `├── README.md`
    * `├── requirements.txt`
    * `├── data/`
        * `└── medical_corpus.txt` (Input text, one document per line)
    * `└── src/`
        * `└── model_trainer.py` (Main script for LDA modeling)
    * `└── notebooks/`
        * `└── exploration.ipynb` (Exploratory notebook)


---

## 🚀 Getting Started

1.  **Clone** and install dependencies:
    ```bash
    git clone (https://github.com/manavj11/Doc-Analysis-v1.git)
    cd Doc-Analysis-v1
    pip install -r requirements.txt
    ```

2.  **Prepare Data**: Ensure your medical documents are in `data/medical_corpus.txt`, with **one document per line**.

3.  **Run**:
    ```bash
    python src/model_trainer.py
    ```

---

## 🔑 Interpreting Results

1. **The Main Script** prints the top words for each found topic. 

2. **The User interprets these words** to manually assign a label within the main script. 

3. **The User Verifies** the model by checking against any one document within the data. 

| Example Topic | Top Words | Assigned Label |
| :---: | :--- | :--- |
| **Topic 1** | drug, trial, dose, patient | **Pharmacology** |
| **Topic 2** | heart, artery, surgery, blood | **Cardiology** |

Hope this gives you an insight into a real world use case of ML!
