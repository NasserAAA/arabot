# Arabot
### FastAPI Zero-Shot Classification Web Service
This web service is built using FastAPI and utilizes the Hugging Face Transformers library for zero-shot text classification. It allows you to classify text inputs into predefined categories without fine-tuning the model.

# Clone the repository:
```
git clone <https://github.com/NasserAAA/arabot>
cd <arabot>
```

# Install project dependencies using Poetry:

```
poetry install
```

# Usage
### Running the Web Service
To start the FastAPI web service, run the following command within your project directory:
```
poetry run uvicorn main:app --reload
```
The service will start on http://localhost:8000 by default.

# API Documentation
Swagger UI: Visit http://localhost:8000/docs in your web browser.

ReDoc: Visit http://localhost:8000/redoc in your web browser.

# Model Explanation
The model used in this web service is `bert-base-uncased` from the Hugging Face Transformers library. It is a pre-trained version of the BERT (Bidirectional Encoder Representations from Transformers) model that has been trained on a large corpus of text data.

## Zero-Shot Classification Approach
Zero-shot classification is achieved by using the `zero-shot-classification` pipeline provided by Hugging Face Transformers with the `bert-base-uncased` model. The pipeline takes a text input and a list of candidate labels and returns the predicted label and confidence scores for each label.

# License
This project is created by **Mohamed Abdelmabood** for **arabot**.
