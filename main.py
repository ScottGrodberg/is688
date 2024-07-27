import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment(text):
    # Analyze the sentiment of the given text
    try:
        sentiment = sia.polarity_scores(text)
    except Exception as e:
        print(e)
        return None    
    return sentiment


# Download the VADER lexicon
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Read the first 500 lines of a CSV file
df = pd.read_csv('dataset.csv', nrows=10000)

# Apply the sentiment analysis to the 'text' column
df['sentiment'] = df['review_text'].apply(analyze_sentiment)

# Expand the sentiment dictionary into separate columns
df = df.join(df['sentiment'].apply(pd.Series))

# Drop the original 'sentiment' column if not needed
df = df.drop(columns=['sentiment'])


grouped = df.groupby(['app_id'])

# Aggregating the grouped data
aggregated = grouped.agg({
    'pos': 'mean',
    'neu': 'mean',
    'neg': 'mean'
})

# Reset index to make app_id columns available for merging
aggregated = aggregated.reset_index()

# Merge the aggregated data back into the original DataFrame
merged_df = pd.merge(df, aggregated, on=['app_id'])

# Display the first few rows of the DataFrame with the sentiment scores
print(merged_df.head())

