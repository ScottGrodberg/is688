import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

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


# Read the first 500 lines of a CSV file
df = pd.read_csv('dataset.csv', nrows=500)

# Apply the sentiment analysis to the 'text' column
df['sentiment'] = df['review_text'].apply(analyze_sentiment)

# Expand the sentiment dictionary into separate columns
df = df.join(df['sentiment'].apply(pd.Series))

# Drop the original 'sentiment' column if not needed
df = df.drop(columns=['sentiment'])

# Display the first few rows of the DataFrame with the sentiment scores
print(df.head())

