#  Chest X-Ray Pneumonia Detection AI
> **An AI-powered diagnostic tool achieving 93.75% accuracy and 96% recall.**

---

##  Project Description
In many clinical settings, radiologists face immense pressure to process high volumes of scans. This project addresses this bottleneck by providing an automated **Binary Classification AI** to identify Pneumonia in pediatric chest X-rays.

###  The "Product" Goal
As a tool designed for medical support, the primary objective was **Reliability**. 
* **Recall-First Strategy:** I prioritized minimizing "False Negatives" (missing a sick patient) because in healthcare, an oversight is more costly than a second check. 
* **Efficiency:** By utilizing **Transfer Learning** (DenseNet121), the model leverages millions of pre-learned visual patterns, making it highly efficient.

---

##  Technical Evolution
* **Phase 1 (Baseline):** Built a CNN from scratch (~59% accuracy).
* **Phase 2 (Transfer Learning):** Implemented DenseNet121 with frozen weights (89% accuracy).
* **Phase 3 (Fine-Tuning):** Unfroze top layers and retrained with a low learning rate (1e-5), reaching the final **93.75%**.

---

##  Performance Metrics
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 93.75% |
| **Recall (Pneumonia)** | 95.90% |
| **Precision (Pneumonia)** | 92.00% |

---

##  Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow, Keras, NumPy, Matplotlib, Seaborn
* **Model:** DenseNet121
* **Platform:** Kaggle
