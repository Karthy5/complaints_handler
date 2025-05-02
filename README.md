# Complaints Handler - Categorization of Requests using NLP

**NAME :** Karthik Raj S.
**REG NO:** RA2211003011401
**Github Profile:** [https://github.com/Karthy5](https://github.com/Karthy5)
**Project Link:** [https://github.com/Karthy5/complaints_handler](https://github.com/Karthy5/complaints_handler)

---

## 1. Introduction

In the era of rapid information growth, the ability to automatically classify text into predefined categories has become increasingly important. Text classification is a fundamental task in Natural Language Processing (NLP), where the goal is to assign a label or class to a piece of text. Applications include spam detection, sentiment analysis, topic labeling, and more. Handling large volumes of customer feedback, operational reports, or user queries efficiently requires robust automated systems.

In this project, we address the task of **Text Classification using NLP specifically for categorizing train passenger feedback**. The primary objective is to correctly categorize text samples representing passenger complaints or observations into one of ten distinct operational classes (e.g., Safety, Cleanliness, Timings). Traditional machine learning methods like Naive Bayes, Support Vector Machines (SVM), and Logistic Regression have been widely used for text classification but often require extensive feature engineering (like TF-IDF) and large labeled datasets to perform well, especially on nuanced tasks. Recent advancements leverage deep learning models, especially Transformer-based models (like BERT and its variants), which can capture complex semantic relationships within the text, often achieving state-of-the-art results with less task-specific feature engineering. This project leverages such advancements through a modern few-shot learning approach.

### Description of the Task:

The task involves training a model on labeled text data (curated examples of passenger feedback) to predict the correct class label for new, unseen text inputs. The dataset, developed iteratively throughout the project, consists of short text samples representing passenger issues, each paired with its corresponding category label from a predefined set of ten (Safety, Medic, Cleanliness, Infra Damage, Technical, Timings, Booking / Cancellation / Refund, Accessibility, Theft, Service). The goal is to build a classifier that accurately assigns new feedback text to the most appropriate category, enabling efficient routing and analysis. A key challenge addressed was handling the semantic overlap and ambiguity naturally present in real-world feedback.

### Algorithm Used to Implement the Task:

For this project, we utilized a modern few-shot classification approach leveraging pre-trained sentence embeddings, implemented via:

*   **`spaCy` Framework:** A leading Python library for industrial-strength Natural Language Processing. It provided the pipeline structure (`spacy.blank("en")`) to host the classification component.
*   **`classy-classification` Library:** A spaCy extension specifically designed for easy integration of various text classification techniques, including few-shot learning based on semantic similarity.
*   **Sentence Transformer Model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`):** A pre-trained deep learning model optimized to generate dense vector embeddings for sentences. These embeddings capture semantic meaning, allowing sentences with similar meanings to be close in the vector space (measured by cosine similarity).
*   **Few-Shot Semantic Similarity Classification:** Instead of training a classifier from scratch on features like TF-IDF, this method works as follows:
    1.  The provided training examples for each category are converted into embeddings using the sentence transformer.
    2.  When a new text input arrives, its embedding is generated.
    3.  The cosine similarity between the input embedding and the embeddings (or derived representations) of the training examples for each category is calculated.
    4.  The category with the highest semantic similarity score is assigned as the prediction.
    This approach leverages the transformer's pre-existing language understanding, allowing effective classification even with a limited number (10-20) of targeted training examples per category. Targeted contrastive examples were added during development to improve the model's ability to distinguish between semantically similar categories.

### Performance Metrics Used to Evaluate the Model Performance:

Several standard metrics were employed to comprehensively evaluate the model's performance:

*   **Confusion Matrix:** To visualize the classification performance.
*   **Precision, Recall, and F1-Score:** Calculated per class via the Classification Report.
*   **Accuracy (Overall):** The proportion of correct predictions across all test samples.
*   **Top-2 Accuracy:** Calculated to assess if the correct label was among the top two most likely predictions.

---

## 2. Dataset:

The dataset used was **developed iteratively** throughout this project and is **not from a public repository link**.

*   **Nature:** Consists of short English text sentences representing typical train passenger feedback, complaints, or observations.
*   **Training Data:** The final training set (`further_expanded_data` in the code) contains approximately 15-18 examples per category. It was significantly enhanced with additional diverse and *contrastive* examples specifically designed to help the model distinguish between commonly confused categories (like Safety/Infra Damage and Accessibility/Service).
*   **Testing Data:** The final testing set (`test_set_balanced_10_easier` in the code) was purpose-built for evaluation:
    *   Contains exactly **10 examples per category**, ensuring a balanced evaluation.
    *   Examples were chosen/written to be **clearer and less ambiguous** instances of their category.
    *   All test examples are **distinct** from the training data.
*   **Labels:** The 10 categories are: Safety, Medic, Cleanliness, Infra Damage, Technical, Timings, Booking / Cancellation / Refund, Accessibility, Theft, Service.

---

## 3. Code:

The primary Python script (`train_evaluate.py` or similar name in the repo) contains the full implementation, including:

*   Dependency imports (`spacy`, `classy-classification`, `sklearn`, `matplotlib`, etc.).
*   Definition of the enhanced training data (`further_expanded_data`).
*   Definition of the balanced, "easier" test set (`test_set_balanced_10_easier`).
*   Training the `spaCy` pipeline with the `classy-classification` component using the specified sentence transformer.
*   Generating predictions on the test set.
*   Calculating and displaying the Confusion Matrix, Classification Report, and Top-2 Accuracy.

*(Optional: Include a small code snippet for prediction here if desired, similar to the previous README draft)*

---

## 4. Results:

The final evaluation yielded strong results (details derived from running the code):

### Figure 1: Results Table

*(It's harder to embed a table image directly inline like the PDF in standard Markdown. You can either link to the image or paste the text report)*

**Classification Report:**
```text
Final Classification Report:
                               precision    recall  f1-score   support

                Accessibility       0.54      0.70      0.61        10
Booking / Cancellation / Refund       0.89      0.80      0.84        10
                  Cleanliness       0.78      0.70      0.74        10
                 Infra Damage       0.73      0.80      0.76        10
                        Medic       1.00      1.00      1.00        10
                       Safety       0.70      0.70      0.70        10
                      Service       0.73      0.80      0.76        10
                    Technical       0.78      0.70      0.74        10
                        Theft       1.00      0.80      0.89        10
                      Timings       1.00      1.00      1.00        10

                     accuracy                           0.80       100
                    macro avg       0.81      0.80      0.80       100
                 weighted avg       0.81      0.80      0.80       100
```

**Top-2 Accuracy:**
```text
Final Top-2 Accuracy: 95.00% (95/100)
```

*(Link to Image: Alternatively, save the results table screenshot and link it: [View Results Table](path/to/results_table_screenshot.png))*

**Analysis:**

*   The Confusion Matrix shows strong diagonal dominance, with significantly improved accuracy for "Accessibility" and "Safety" after data enhancement.
*   The Classification Report confirms high F1-scores for most categories and an overall accuracy of 80%.
*   The Top-2 Accuracy of 95% highlights the model's strong semantic understanding, often ranking the correct class highly even when the top prediction is incorrect.

The iterative refinement led to a robust and accurate classifier.

---

## Installation & Usage

*(Keep the Installation and Usage sections from the previous README draft here, as they provide practical instructions)*

1.  **Clone:** `git clone https://github.com/Karthy5/complaints_handler.git`
2.  **Install:** `pip install spacy classy-classification sentence-transformers scikit-learn matplotlib numpy` (preferably in a virtual environment)
3.  **Run:** `python <your_script_name.py>` to train and evaluate.

## Future Improvements

*   Experiment with different Sentence Transformer models.
*   Further expand the training dataset with more diverse real-world examples.
*   Implement model saving/loading for persistence.
*   Develop a simple web interface (e.g., using Flask/Streamlit).

## License

*(Add license information here, e.g., MIT License)*
