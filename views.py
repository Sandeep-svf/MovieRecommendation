from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
# Create your views here.

def index(request):
    return render(request,'index.html')

def color(request):
    return render(request,'top.html')

def reccom(request):
    import pandas as pd
    import numpy as np
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    from scipy import stats
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.corpus import wordnet
    from surprise import Reader, Dataset, SVD, evaluate

    import warnings; warnings.simplefilter('ignore')


    md = pd. read_csv('/home/user/akku/movies_metadata.csv')
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()#mean rating
    m = md['vote_count'].quantile(0.90)#Calculate the minimum number of votes required to be in the chart, m
    q_movies = md.copy().loc[md['vote_count'] >= m]# Filter out all qualified movies into a new DataFrame
    def weighted_rating(x):
        # Function that computes the weighted rating of each movie
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)
    #Define a new feature 'score' and calculate its value with `weighted_rating()
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
    #Finally, let's sort the DataFrame based on the score feature and output the title, vote count, vote average and weighted rating or score of the top 250 movies.
    q_movies = q_movies.sort_values('score', ascending=False)
    #Print the top 250 movies
    x=q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15)
    y=x.to_html()
    response=y
    return HttpResponse(response)

def content(request):
    
        import pandas as pd
        import numpy as np
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        from scipy import stats
        from ast import literal_eval
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
        from nltk.stem.snowball import SnowballStemmer
        from nltk.stem.wordnet import WordNetLemmatizer
        from nltk.corpus import wordnet
        from surprise import Reader, Dataset, SVD, evaluate

        import warnings; warnings.simplefilter('ignore')


        md = pd. read_csv('/home/user/akku/movies_metadata.csv')
        links_small = pd.read_csv('/home/user/akku/links_small.csv')
        links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
        md = md.drop([19730, 29503, 35587])
        #Check EDA Notebook for how and why I got these indices.
        md['id'] = md['id'].astype('int')
        smd = md[md['id'].isin(links_small)]
        smd['tagline'] = smd['tagline'].fillna('')
        smd['description'] = smd['overview'] + smd['tagline']
        smd['description'] = smd['description'].fillna('')
        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(smd['description'])

        #You will compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document. This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document) and each column represents a movie, as before.
        #In its essence, the TF-IDF score is the frequency of a word occurring in a document, down-weighted by the number of documents in which it occurs. This is done to reduce the importance of words that occur frequently in plot overviews and therefore, their significance in computing the final similarity score
        #Import TfIdfVectorizer from scikit-learn

        #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a

        #Replace NaN with an empty string

        #Construct the required TF-IDF matrix by fitting and transforming the da

        #Output the shape of tfidf_matrix
        tfidf_matrix.shape
        #With this matrix in hand, you can now compute a similarity score
        #using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. You use the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate (especially when used in conjunction with TF-IDF scores,
        #Since you have used the TF-IDF vectorizer, calculating the dot product will directly give you the cosine similarity score. Therefore, you will use sklearn's linear_kernel() instead of cosine_similarities() since it is faster.
        # Import linear_kernel
        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        #You're going to define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies. Firstly, for this, you need a reverse mapping of movie titles and DataFrame indices. In other words, you need a mechanism to identify the index of a movie in your metadata DataFrame, given its title.
        #Construct a reverse map of indices and movie title
        smd = smd.reset_index()
        titles = smd['title']
        indices = pd.Series(smd.index, index=smd['title'])
        # Function that takes in movie title as input and outputs most similar movies
        def get_recommendations(title):
            
            # Get the index of the movie that matches the title
            idx = indices[title]
            # Get the scores of the 10 most similar movies
            # Get the pairwsie similarity scores of all movies with that movie

            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:31]
             # Get the movie indices
            movie_indices = [i[0] for i in sim_scores]
            return titles.iloc[movie_indices]
        name = request.GET['s1']
        
        # Return the top 10 most similar movies
        y=get_recommendations(name).head(10)
        s = "<table> <tr> <td><b>Movie Names</b></td> </tr>" 
        for index, value in y.iteritems():
            s = s + "<tr><td> " + value + " </td></tr>"
        s = s + "</table>"
        return HttpResponse(s)

def Romance(request):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

    import warnings; warnings.simplefilter('ignore')

    md = pd. read_csv('/home/user/akku/movies_metadata.csv')
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.90) #calculates the minimum no of votes in 90th percentile
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    qualified = (md.copy().loc[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average','popularity', 'genres']])
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)  #imdb weighted  rating formula
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(15)
    s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = md.drop('genres', axis=1).join(s)
    def build_chart(genre, percentile=0.85):
        df = gen_md[gen_md['genre'] == genre]
        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(250)
        return qualified
    x=build_chart('Romance').head(15)
    y=x.to_html()
    return HttpResponse(y)


def Action(request):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

    import warnings; warnings.simplefilter('ignore')

    md = pd. read_csv('/home/user/akku/movies_metadata.csv')
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.90) #calculates the minimum no of votes in 90th percentile
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    qualified = (md.copy().loc[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average','popularity', 'genres']])
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)  #imdb weighted  rating formula
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(15)
    s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = md.drop('genres', axis=1).join(s)
    def build_chart(genre, percentile=0.85):
        df = gen_md[gen_md['genre'] == genre]
        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(250)
        return qualified
    x=build_chart('Action').head(15)
    y=x.to_html()
    return HttpResponse(y)


def Mystery(request):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

    import warnings; warnings.simplefilter('ignore')

    md = pd. read_csv('/home/user/akku/movies_metadata.csv')
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.90) #calculates the minimum no of votes in 90th percentile
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    qualified = (md.copy().loc[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average','popularity', 'genres']])
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)  #imdb weighted  rating formula
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(15)
    s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = md.drop('genres', axis=1).join(s)
    def build_chart(genre, percentile=0.85):
        df = gen_md[gen_md['genre'] == genre]
        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(250)
        return qualified
    x=build_chart('Mystery').head(15)
    y=x.to_html()
    return HttpResponse(y)

def Animated(request):
    import pandas as pd
    import numpy as np
    from scipy import stats
    from ast import literal_eval
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

    import warnings; warnings.simplefilter('ignore')

    md = pd. read_csv('/home/user/akku/movies_metadata.csv')
    md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.90) #calculates the minimum no of votes in 90th percentile
    md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    qualified = (md.copy().loc[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average','popularity', 'genres']])
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)  #imdb weighted  rating formula
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(15)
    s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'genre'
    gen_md = md.drop('genres', axis=1).join(s)
    def build_chart(genre, percentile=0.85):
        df = gen_md[gen_md['genre'] == genre]
        vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
        vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
        C = vote_averages.mean()
        m = vote_counts.quantile(percentile)

        qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
        qualified['vote_count'] = qualified['vote_count'].astype('int')
        qualified['vote_average'] = qualified['vote_average'].astype('int')

        qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
        qualified = qualified.sort_values('wr', ascending=False).head(250)
        return qualified
    x=build_chart('Animation').head(15)
    y=x.to_html()
    return HttpResponse(y)

    
    




