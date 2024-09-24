import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download necessary NLTK data files
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize sentiment analyzer and stop words
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load reviews from CSV
def load_reviews():
    return pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    VALID_LOCATIONS = [
        "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
        "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
        "El Paso, Texas", "Escondido, California", "Fresno, California",
        "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
        "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
        "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
    ]

    def __init__(self) -> None:
        self.reviews = load_reviews()

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            query_string = environ.get("QUERY_STRING", "")
            params = parse_qs(query_string)

            location = params.get('location', [None])[0]
            start_date = params.get('start_date', [None])[0]
            end_date = params.get('end_date', [None])[0]

            # Filter reviews by location
            filtered_reviews = self.reviews
            if location:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            # Filter reviews by date range
            if start_date or end_date:
                try:
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                    if end_date:
                        end_date = pd.to_datetime(end_date)

                    filtered_reviews = [
                        review for review in filtered_reviews
                        if (start_date is None or pd.to_datetime(review['Timestamp']) >= start_date) and
                           (end_date is None or pd.to_datetime(review['Timestamp']) <= end_date)
                    ]
                except ValueError:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [b'{"error": "Invalid date format"}']

            # Prepare results with sentiment analysis
            results = []
            for review in filtered_reviews:
                sentiment = self.analyze_sentiment(review['ReviewBody'])
                review_data = {
                    "ReviewId": review['ReviewId'],
                    "ReviewBody": review['ReviewBody'],
                    "Location": review['Location'],
                    "Timestamp": review['Timestamp'],
                    "sentiment": sentiment
                }
                results.append(review_data)

            # Sort results by compound sentiment score in descending order
            results.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(results, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            post_data = environ["wsgi.input"].read(content_length).decode("utf-8")
            params = parse_qs(post_data)

            location = params.get('Location', [None])[0]
            review_body = params.get('ReviewBody', [None])[0]

            # Validate input
            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [b'{"error": "Location and ReviewBody are required"}']

            if location not in self.VALID_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [b'{"error": "Invalid location"}']

            # Create a new review
            review_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            new_review = {
                "ReviewId": review_id,
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": timestamp
            }

            # Append the new review to the in-memory list
            self.reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()