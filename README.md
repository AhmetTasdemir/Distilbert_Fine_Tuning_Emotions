# Fine-Tuned DistilBERT Model for Emotion Classification

This repository contains a notebook for fine-tuning the DistilBERT model on the emotion classification task. The trained model is available on the Hugging Face Model Hub.

**Fine-Tuned Model Link**: [DistilBERT Base Uncased Fine-Tuned Emotion](https://huggingface.co/ahmettasdemir/distilbert-base-uncased-finetuned-emotion)

## Dataset

The emotion classification task is performed on the emotions dataset, which contains textual data labeled with different emotions. The dataset is divided into three subsets:

- **Train**: Used for training the model (16,000 samples)
- **Validation**: Used for evaluating the model during training (2,000 samples)
- **Test**: Used for evaluating the final model performance (2,000 samples)

The dataset is available through the [Hugging Face Datasets](https://huggingface.co/datasets) library.

## Contents

The notebook is divided into the following sections:

1. **Data Loading and Exploration**: Loading the dataset and exploring its structure, including the number of samples, column names, and sample texts.
2. **From Datasets to DataFrames**: Converting the dataset to pandas DataFrames for easier data manipulation and visualization.
3. **Class Distribution Analysis**: Analyzing the class distribution of the dataset using bar plots.
4. **Text Length Analysis**: Analyzing the length of the tweets in terms of the number of words.
5. **Tokenization**: Tokenizing the text data using the DistilBERT tokenizer.
6. **Model Initialization and Configuration**: Initializing the pre-trained DistilBERT model for sequence classification and configuring the training settings.
7. **Training and Evaluation**: Fine-tuning the model on the training dataset and evaluating its performance on the validation dataset.
8. **Evaluation Metrics**: Computing evaluation metrics such as accuracy and F1 score on the test dataset.
9. **Conclusion**: A summary of the notebook's contents and the process of fine-tuning the DistilBERT model for emotion classification.

## Usage

To use the fine-tuned DistilBERT model for emotion classification, follow these steps:

1. Install the required dependencies listed in the `requirements.txt` file.
2. Load the trained model using the Hugging Face Transformers library.
3. Tokenize the input text using the DistilBERT tokenizer.
4. Pass the tokenized input through the model to get the predicted emotions.

```python
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_name = "ahmettasdemir/distilbert-base-uncased-finetuned-emotion"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the text classification pipeline
pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Example text to classify
text = "I'm feeling happy today!"

# Perform text classification
result = pipeline(text)

# Print the predicted label
predicted_label = result[0]['label']
print("Predicted Emotion:", predicted_label)
