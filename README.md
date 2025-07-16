# Sentiment Analysis of Amazon Fine Food Reviews

## Project Overview
This project focuses on performing sentiment analysis on the Amazon Fine Food Reviews dataset using a Long Short-Term Memory (LSTM) neural network. The goal is to classify reviews as positive or negative based on their text content. The model is built using TensorFlow and leverages natural language processing (NLP) techniques such as tokenization, stemming, and word embeddings to preprocess and analyze the text data. The project achieves approximately 80% accuracy, demonstrating robust performance for sentiment classification tasks.

## Dataset
The dataset used is the [Sentiment140 dataset](http://help.sentiment140.com/), containing 1.6 million tweets labeled as positive (4) or negative (0). Each entry includes:
- **Sentiment**: 0 (negative) or 4 (positive)
- **ID**: Unique tweet identifier
- **Date**: Timestamp of the tweet
- **Query**: Query used (NO_QUERY in this dataset)
- **User ID**: Twitter user handle
- **Text**: The tweet content

The dataset is preprocessed to clean text (removing URLs, mentions, special characters, etc.) and prepare it for training the LSTM model.

## Project Structure
- `nlp-text-classification-lstm.ipynb`: Jupyter Notebook containing the complete code for data preprocessing, model building, training, and evaluation.
- `README.md`: This file, providing an overview and instructions for the project.

## Requirements
To run this project, you need the following dependencies:
- Python 3.10+
- TensorFlow 2.15.0
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- Jupyter Notebook

You can install the required packages using:
```bash
pip install tensorflow pandas numpy nltk scikit-learn matplotlib jupyter
```

Additionally, download the NLTK stopwords dataset within the notebook:
```python
import nltk
nltk.download('stopwords')
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Sentiment-Analysis-of-Amazon-Fine-Food-Reviews.git
   cd Sentiment-Analysis-of-Amazon-Fine-Food-Reviews
   ```
2. Install the required dependencies (see above).
3. Download the Sentiment140 dataset and place it in the appropriate directory (e.g., `/kaggle/input/sentiment140/` in the notebook). Alternatively, update the file path in the notebook to match your local setup.

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook nlp-text-classification-lstm.ipynb
   ```
2. Run the cells in the notebook sequentially to:
   - Load and preprocess the dataset
   - Build and train the LSTM model
   - Evaluate the model using a confusion matrix and classification report
3. The notebook includes visualizations (e.g., confusion matrix) and a classification report showing precision, recall, and F1-score for the model.

## Model Details
- **Architecture**: The model uses an LSTM network, which is well-suited for sequential data like text. It includes:
  - Embedding layer to convert words into dense vectors
  - LSTM layer to capture long-term dependencies in text
  - Dense layers for classification
- **Preprocessing**:
  - Text cleaning using regex to remove URLs, mentions, and special characters
  - Stopword removal and stemming using NLTK
  - Tokenization and padding to ensure uniform input length
- **Performance**: The model achieves ~79% accuracy, with balanced precision and recall for both positive and negative classes.

## Results
The model was evaluated on a test set, producing the following classification report:
```
              precision    recall  f1-score   support
    Negative       0.79      0.78      0.79    160542
    Positive       0.78      0.79      0.79    159458
    accuracy                           0.79    320000
   macro avg       0.79      0.79      0.79    320000
weighted avg       0.79      0.79      0.79    320000
```
The confusion matrix visualization is included in the notebook to illustrate the model's performance in distinguishing between positive and negative sentiments.

## Limitations
- The model is trained on English tweets, so non-English text may not be processed effectively.
- The dataset may contain noisy or ambiguous sentiments, which could affect model performance.
- Further hyperparameter tuning or advanced architectures (e.g., BERT) could potentially improve accuracy.

## Future Improvements
- Incorporate advanced NLP models like BERT or RoBERTa for better performance.
- Handle non-English tweets by adding multilingual support.
- Experiment with hyperparameter tuning (e.g., learning rate, LSTM units) to optimize the model.
- Add cross-validation to ensure robustness across different data splits.

## Contributing
Contributions are welcome! If you have suggestions or improvements:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The Sentiment140 dataset creators for providing the data.
- TensorFlow and NLTK communities for their excellent libraries and documentation.



You can copy the content within the `<xaiArtifact>` tags (excluding the tags themselves) and paste it directly into your `README.md` file on GitHub. Make sure to replace `your-username` in the clone command with your actual GitHub username.
