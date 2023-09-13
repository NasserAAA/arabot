from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from predict import predict_label_with_confidence

app = FastAPI()


class ClassificationRequest(BaseModel):
    """
    Request model for text classification.
    """
    text: str
    candidate_labels: List[str]


class BatchClassificationRequest(BaseModel):
    """
    Request model for batch text classification.
    """
    texts: List[str]
    candidate_labels: List[str]


class ClassificationResponse(BaseModel):
    """
    Response model for text classification.
    """
    predicted_label: str
    confidence_scores: dict


class BatchClassificationResponse(BaseModel):
    """
    Response model for batch text classification.
    """
    predictions: List[dict]


@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    """
    Classify the input text and return the predicted label and confidence scores.

    Args:
        request (ClassificationRequest): Request object containing text and candidate labels.

    Returns:
        ClassificationResponse: Response object containing predicted label and confidence scores.
    """
    text = request.text
    candidate_labels = request.candidate_labels

    try:
        label_confidence_pairs = predict_label_with_confidence(text, candidate_labels)

        if not label_confidence_pairs:
            # Handle the case where no predictions are available
            raise HTTPException(status_code=404, detail="No prediction available")

        # Find the label with the highest confidence score
        max_confidence_idx = label_confidence_pairs.index(max(label_confidence_pairs, key=lambda x: x['confidence']))
        predicted_label = label_confidence_pairs[max_confidence_idx]['label']

        # Create a dictionary with labels as keys and numerical scores as values
        confidence_scores_dict = {pair['label']: pair['confidence'] for pair in label_confidence_pairs}

        return {"predicted_label": predicted_label, "confidence_scores": confidence_scores_dict}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/classify_batch", response_model=BatchClassificationResponse)
async def classify_text_batch(request: BatchClassificationRequest):
    """
    Classify a batch of input texts and return a list of predicted labels and confidence scores.

    Args:
        request (BatchClassificationRequest): Request object containing a list of texts and candidate labels.

    Returns:
        BatchClassificationResponse: Response object containing a list of predicted labels and confidence scores.
    """
    texts = request.texts
    candidate_labels = request.candidate_labels

    try:
        batch_predictions = []

        for text in texts:
            label_confidence_pairs = predict_label_with_confidence(text, candidate_labels)

            if not label_confidence_pairs:
                # Handle the case where no predictions are available for a text
                prediction = {"predicted_label": "No prediction available", "confidence_scores": {}}
            else:
                # Find the label with the highest confidence score
                max_confidence_idx = label_confidence_pairs.index(
                    max(label_confidence_pairs, key=lambda x: x['confidence']))
                predicted_label = label_confidence_pairs[max_confidence_idx]['label']

                # Create a dictionary with labels as keys and numerical scores as values
                confidence_scores_dict = {pair['label']: pair['confidence'] for pair in label_confidence_pairs}
                prediction = {"text": text, "predicted_label": predicted_label, "confidence_scores": confidence_scores_dict}

            batch_predictions.append(prediction)

        return {"predictions": batch_predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
