import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Step 1: Load your cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Step 2: Vectorize the tags column (convert text into numbers)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# Step 3: Calculate similarity between movies
similarity = cosine_similarity(vectors)

# Step 4: Recommendation function
def recommend(movie):
    if movie not in df['title'].values:
        print("Movie not found 😞")
        return
    index = df[df['title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6] # Top 5
    print(f"\nMovies similar to '{movie}':\n")
    for i in movies_list:
        print(df.iloc[i[0]].title)

# Step 5: Try it out!
recommend("Iron Man") # You can change the movie title here

# Step 6 (Optional): Save model for future use
pickle.dump(df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
