import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
def analyze_sentiment(text):
    # Analyze the sentiment of the given text
    sentiment = sia.polarity_scores(text)
    return sentiment


# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example usage
steam_review = "This game is amazing! The graphics are stunning and the gameplay is addictive."
sentiment = analyze_sentiment(steam_review)


