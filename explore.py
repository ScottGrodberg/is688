import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Basic information about the dataset
print("Basic Information:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Summary statistics of the dataset
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Distribution of review scores
plt.figure(figsize=(8, 6))
sns.countplot(x='review_score', data=df, palette='coolwarm')
plt.title('Distribution of Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Count')
plt.show()

# Distribution of review votes
plt.figure(figsize=(8, 6))
sns.histplot(df['review_votes'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Review Votes')
plt.xlabel('Review Votes')
plt.ylabel('Frequency')
plt.show()

# Proportion of early access reviews
early_access_reviews = df['app_name'].isnull().sum() / len(df) * 100
print(f"\nPercentage of missing game names: {early_access_reviews:.2f}%")

# Proportion of early access reviews
early_access_proportion = len(df[df['review_text'].str.contains('early access', case=False, na=False)]) / len(df) * 100
print(f"Percentage of early access reviews: {early_access_proportion:.2f}%")

# Most reviewed games
most_reviewed_games = df['app_name'].value_counts().head(10)
print("\nMost reviewed games:")
print(most_reviewed_games)

# Plot most reviewed games
plt.figure(figsize=(12, 8))
sns.barplot(x=most_reviewed_games.values, y=most_reviewed_games.index, palette='viridis')
plt.title('Top 10 Most Reviewed Games')
plt.xlabel('Number of Reviews')
plt.ylabel('Game Name')
plt.show()

# Word cloud of review texts
from wordcloud import WordCloud

review_text = ' '.join(df['review_text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Review Texts')
plt.show()
