import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import random

def analyze_sentiment(text):
    # Analyze the sentiment of the given text
    try:
        sentiment = sia.polarity_scores(text)
    except Exception as e:
        print(e)
        return None    
    return sentiment


def read_csv_random_sample(file_path, sample_size):
    # Get the number of lines in the file (excluding header)
    with open(file_path, 'r') as file:
        row_count = sum(1 for row in file) - 1  # Subtract 1 for the header

    # Generate a sorted list of random line numbers to skip
    skip_lines = sorted(random.sample(range(1, row_count + 1), row_count - sample_size))


    # Read the sampled data
    df_sample = pd.read_csv(file_path, skiprows=skip_lines)

    return df_sample


# Download the VADER lexicon
# nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()


# Read the first n lines of a CSV file
df = read_csv_random_sample('dataset.csv', 10)

# For each record, apply the sentiment analysis to the 'text' column
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

deduplicated_df = df.drop_duplicates(subset=['app_name'])

print(deduplicated_df.head())

deduplicated_df.to_csv('dataset_w_sentiment_by_app_name.csv', index=False)


