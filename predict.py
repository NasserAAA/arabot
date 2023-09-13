from transformers import pipeline
import logging


def predict_label_with_confidence(text, candidate_labels, model_name="bert-base-uncased"):
    """
    Predict labels with confidence scores using zero-shot classification.

    Args:
        text (str): The input text for classification.
        candidate_labels (list): List of candidate labels to predict from.
        model_name (str): The name of the pre-trained model to use.

    Returns:
        list: A list of dictionaries containing label-confidence pairs.
    """
    try:
        classifier_pipeline = pipeline("zero-shot-classification", model=model_name)

        result = classifier_pipeline(text, candidate_labels)

        labels = result['labels']
        scores = result['scores']

        label_confidence_pairs = [{'label': label, 'confidence': score} for label, score in zip(labels, scores)]

        return label_confidence_pairs
    except Exception as e:
        logging.error(f"Error in predict_label_with_confidence: {str(e)}")
        return []
