import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie data (assuming 'movies.csv' file exists)
movies_df = pd.read_csv('movies.csv')

# Function to get movie recommendations
def get_movie_recommendations(movie_name, top_n=10):
    # Filter movies that contain the specified name
    movie_details = movies_df[movies_df['title'].str.contains(movie_name, case=False)]

    # Check if the movie is not found
    if movie_details.empty:
        return f"No movies found with a name similar to '{movie_name}'."

    # Vectorize movie genres using CountVectorizer
    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform(movie_details['genres'])

    # Calculate cosine similarity between movies based on genres
    similarity_scores = cosine_similarity(genre_matrix, genre_matrix)

    # Get the index of the specified movie
    movie_index = movie_details.index[0]

    # Check if the movie index is valid
    if movie_index >= len(similarity_scores) or movie_index < 0:
        return "Error: Invalid movie index."

    # Get similarity scores for the specified movie
    similar_movies_scores = similarity_scores[movie_index]

    # Create a DataFrame with movie titles and their similarity scores
    similar_movies_df = pd.DataFrame({
        'title': movie_details['title'],
        'similarity_score': similar_movies_scores
    })

    # Sort the DataFrame by similarity scores in descending order
    similar_movies_df = similar_movies_df.sort_values(by='similarity_score', ascending=False)

    # Exclude the specified movie from the recommendations
    similar_movies_df = similar_movies_df[similar_movies_df.index != movie_index]

    # Get the top recommendations
    top_recommendations = similar_movies_df.head(top_n)

    # Return the top recommendations
    return top_recommendations

# User Input Section
movie_name = input("Enter a movie name to get recommendations: ")

# Get Recommendations Section
recommendations = get_movie_recommendations(movie_name)

# Display Recommendations Section
if isinstance(recommendations, str):
    print(recommendations)
else:
    print(f"Top {len(recommendations)} Movie Recommendations for '{movie_name}':")
    print(recommendations[['title', 'similarity_score']])
