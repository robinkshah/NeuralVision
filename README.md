
# Brain Tumour Detection using Deep Learning

This project utilizes Convolutional Neural Networks (CNNs) to automate the detection and classification of brain tumours from MRI scans. The goal is to support medical diagnosis by identifying tumour presence and type with high accuracy using AI-driven image analysis.

## 🧠 Project Overview

- **Goal**: Classify MRI images into tumour types (e.g., glioma, meningioma, pituitary) or identify absence of tumour.
- **Dataset**: MRI images sourced from publicly available datasets.
- **Output**: Multi-class classification (Tumour Type or No Tumour).

## 🧪 Deep Learning Architecture

- **Preprocessing**:
  - Image resizing, normalization, and augmentation.
- **Model Architectures**:
  - Transfer learning using models like VGG16, InceptionV3, ResNet50
  - Custom CNN designed for image classification
- **Training**:
  - Optimized using Adam optimizer and categorical cross-entropy loss.
  - Validation using stratified k-fold cross-validation.
- **Evaluation**:
  - Accuracy, precision, recall, F1-score
  - Confusion matrix and ROC curves

## 📂 Project Structure

```
Brain_Tumour_Detection_AI/
├── data/                    # MRI scan dataset
├── models/                  # Saved models and weights
├── results/                 # Evaluation metrics and visualizations
└── Brain_Tumour_Detection_AI.ipynb  # Main Jupyter Notebook
```

## 📈 Results Summary

| Model         | Accuracy | F1 Score |
|---------------|----------|----------|
| VGG16         | 95.2%    | 0.95     |
| ResNet50      | 96.8%    | 0.96     |
| InceptionV3   | 97.5%    | 0.97     |
| Custom CNN    | 94.1%    | 0.94     |

*(Example results; replace with actual values if different)*

## 🛠️ Tools & Libraries

- Python, TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Scikit-learn

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumour-detection-ai.git
   cd brain-tumour-detection-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook Brain_Tumour_Detection_AI.ipynb
   ```

4. Run the cells and test predictions using sample images.

## 🧬 Ethical Considerations

This tool is intended for **research and educational use only**. It is **not a substitute for professional medical diagnosis**.

## 👤 Author

**Robin Shah**  
BSc. in Computer Science | AI & Healthcare Researcher  
[LinkedIn](https://linkedin.com/in/robinkshah) | [GitHub](https://github.com/robinkshah)

---

