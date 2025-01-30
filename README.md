# End-to-End Sentiment Analysis Pipeline

## Objective

This project implements a sentiment analysis pipeline using the IMDB Movie Reviews dataset. The goal is to build a model that classifies movie reviews as positive or negative and serve predictions via a Flask API.

## Dataset

- **IMDB Movie Reviews Dataset**
- Available on:
  - Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


## Environment Setup

### Prerequisites

- Python 3.x
- Virtual environment (recommended)

### Installation Steps

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <project-directory>
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Flask Application

### Train the Model

Run the training script to train and save the sentiment analysis model:

```sh
python train.py
```

### Start the Flask API Server

Run the following command to start the server:

```sh
python app.py
```

The API will be available at `http://127.0.0.1:5000/`.

### Testing the API

Use `curl`, Postman, or Python to send a test request:

```sh
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"review_text": "This movie was amazing!"}'
```

Or using Python:

```python
import requests
url = "http://127.0.0.1:5000/predict"
data = {"review_text": "This movie was amazing!"}
response = requests.post(url, json=data)
print(response.json())
```


## Contact

For any queries, reach out to *[gautamaman085@gmail.com]*.

---

Feel free to modify any sections to better suit your project structure!

