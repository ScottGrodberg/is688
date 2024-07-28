import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

def analyze_sentiment(text):
    try:
        sentiment = sia.polarity_scores(text)
    except Exception as e:
        print(e)
        return None    
    return sentiment

def read_csv_random_sample(file_path, sample_size):
    with open(file_path, 'r') as file:
        row_count = sum(1 for row in file) - 1  # Subtract 1 for the header

    skip_lines = sorted(random.sample(range(1, row_count + 1), row_count - sample_size))

    df_sample = pd.read_csv(file_path, skiprows=skip_lines)

    return df_sample

# Download the VADER lexicon if not already downloaded
# nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Read a random sample from the CSV file
df = read_csv_random_sample('dataset.csv', 10)

# Apply sentiment analysis to the 'review_text' column
df['sentiment'] = df['review_text'].apply(analyze_sentiment)

# Expand the sentiment dictionary into separate columns
df = df.join(df['sentiment'].apply(pd.Series))

# Drop the original 'sentiment' column
df = df.drop(columns=['sentiment'])

# Group and aggregate sentiment scores by 'app_id'
grouped = df.groupby(['app_id'])

aggregated = grouped.agg({
    'pos': 'mean',
    'neu': 'mean',
    'neg': 'mean'
})

aggregated = aggregated.reset_index()

# Merge the aggregated data back into the original DataFrame
merged_df = pd.merge(df, aggregated, on=['app_id'], suffixes=('', '_mean'))

# Deduplicate data based on 'app_name'
deduplicated_df = df.drop_duplicates(subset=['app_name'])

# Save the processed data to a CSV file
deduplicated_df.to_csv('dataset_w_sentiment_by_app_name.csv', index=False)

# Visualization 1: Distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df['compound'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Review Sentiment Scores')
plt.xlabel('Compound Sentiment Score')
plt.ylabel('Number of Reviews')
plt.axvline(df['compound'].mean(), color='red', linestyle='dashed', linewidth=1)
plt.text(df['compound'].mean(), plt.ylim()[1]*0.9, f'Mean: {df["compound"].mean():.2f}', color = 'red')
plt.show()

# Visualization 2: Average sentiment scores by app_id
plt.figure(figsize=(12, 8))
aggregated_melted = aggregated.melt(id_vars=['app_id'], value_vars=['pos', 'neu', 'neg'],
                                    var_name='Sentiment', value_name='Score')
sns.barplot(data=aggregated_melted, x='app_id', y='Score', hue='Sentiment', palette='viridis')
plt.title('Average Sentiment Scores by App ID')
plt.xlabel('App ID')
plt.ylabel('Average Sentiment Score')
plt.legend(title='Sentiment Type', loc='upper right')
plt.xticks(rotation=45)
plt.show()

# Visualization 3: Average sentiment scores by app_name
plt.figure(figsize=(12, 8))
app_name_aggregated = deduplicated_df.groupby('app_name').agg({
    'pos': 'mean',
    'neu': 'mean',
    'neg': 'mean'
}).reset_index()

app_name_aggregated_melted = app_name_aggregated.melt(id_vars=['app_name'], value_vars=['pos', 'neu', 'neg'],
                                                      var_name='Sentiment', value_name='Score')
sns.barplot(data=app_name_aggregated_melted, x='app_name', y='Score', hue='Sentiment', palette='coolwarm')
plt.title('Average Sentiment Scores by App Name')
plt.xlabel('App Name')
plt.ylabel('Average Sentiment Score')
plt.legend(title='Sentiment Type', loc='upper right')
plt.xticks(rotation=45)
plt.show()

# Topic Modeling using TF-IDF and LDA
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['review_text'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

num_top_words = 10
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
display_topics(lda, tfidf_feature_names, num_top_words)

# Visualization 4: Word Clouds for Each Topic
for topic_idx, topic in enumerate(lda.components_):
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(stopwords='english', background_color='white').generate(
        " ".join([tfidf_feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Topic {topic_idx + 1}")
    plt.show()
