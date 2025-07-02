import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Recommend function
def recommend(movie):
    movie = movie.lower()
    movie_index = movies[movies['title'].str.lower() == movie].index
    if movie_index.empty:
        return ["Movie not found 😞"]
    index = movie_index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [movies.iloc[i[0]].title for i in distances]
    return recommended_movies

# Streamlit UI
st.title('🎬 Movie Recommender System')
selected_movie = st.text_input('Type a movie name (e.g., Inception)')

if st.button('Show Recommendations'):
    recommendations = recommend(selected_movie)
    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")
        