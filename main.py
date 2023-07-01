# This is a sample Python script.
import numpy as np
from PIL._imaging import display

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
import gensim

from sklearn.metrics import f1_score


import nltk

from fuzzywuzzy import fuzz

import Levenshtein

import pandas as pd
import spacy

import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from scipy.stats import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, precision_score

from sklearn.metrics.pairwise import cosine_similarity

# Press the green button in the gutter to run the script.
# nlp = spacy.load('en_core_web_md')
nltk.download('stopwords')
nltk.download('punkt')

def calculate_f1(df_actual, df_predicted):
    # Flatten the DataFrame values to handle multi-index columns if present
    actual_values = df_actual.values.flatten()
    predicted_values = df_predicted.values.flatten()

    # Calculate precision
    precision = precision_score(actual_values, predicted_values, average='binary')

    # Calculate recall
    recall = recall_score(actual_values, predicted_values, average='binary')

    # Calculate F1 score
    f1_result = f1_score(actual_values, predicted_values, average='binary')

    return precision, recall, f1_result


def apply_selector(df):
    selector_df = pd.DataFrame(np.where(df >= 0.5, 1, 0), columns=df.columns, index=df.index)
    return selector_df


def apply_selector(df):
    selector_df = pd.DataFrame(np.where(df >= 0.5, 1, 0), columns=df.columns, index=df.index)
    return selector_df


class DataFrameCombiner:
    def __init__(self, df1, df2, df3):
        self.df1 = df1
        self.df2 = df2
        self.df3 = df3

    def get_max_dataframe(self):
        numeric_columns = self.df1.select_dtypes(include=[np.number]).columns

        max_df = pd.DataFrame(np.maximum(self.df1[numeric_columns].values,
                                         self.df2[numeric_columns].values,
                                         self.df3[numeric_columns].values),
                              columns=numeric_columns,
                              index=self.df1.index)

        return max_df


class DataFrameCombiner_Average:
    def __init__(self, df1, df2, df3):
        self.df1 = df1
        self.df2 = df2
        self.df3 = df3

    def get_average_dataframe(self):
        numeric_columns = self.df1.select_dtypes(include=[np.number]).columns

        average_df = pd.DataFrame((self.df1[numeric_columns].values +
                                   self.df2[numeric_columns].values +
                                   self.df3[numeric_columns].values) / 3,
                                  columns=numeric_columns,
                                  index=self.df1.index)

        return average_df


class DataFrameCombiner_MA:
    def __init__(self, df1, df2, df3, weights):
        self.df1 = df1
        self.df2 = df2
        self.df3 = df3
        self.weights = weights

    def get_weighted_average_dataframe(self):
        numeric_columns = self.df1.select_dtypes(include=[np.number]).columns

        weighted_average_df = pd.DataFrame((self.df1[numeric_columns].values * self.weights[0] +
                                            self.df2[numeric_columns].values * self.weights[1] +
                                            self.df3[numeric_columns].values * self.weights[2]) / np.sum(self.weights),
                                           columns=numeric_columns,
                                           index=self.df1.index)

        return weighted_average_df


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

mediated_scheme = ['Name', 'Last Name', 'First Name', 'Team', 'Team name', 'Club', 'Country', 'Age', 'Birthday', 'B_Day'
                                                                                                                 'Gender',
                   'Salary',
                   'Year', 'Joined year']

nba = ['Name', 'Rating', 'Jersey', 'Team', 'Position', 'B_Day', 'Height', 'Weight', 'Salary', 'Country',
       'BestPlayerOfTheYear', 'College', 'Version'
       ]

cycling = ['Year', 'Function', 'Last Name', 'First Name', 'Birth date', 'B_date_US', 'Age',
           'Gender', 'Category', 'Country', 'Continent', 'Team_Code', 'Team_Name', 'UCIID', 'Name'
           ]

fifa21 = ['ID', 'Name', 'Age', 'Country', 'Club', 'Position', 'Height', 'Weight', 'foot', 'Joined', 'Salary',
          'Contract Date', 'Finishing', 'Heading', 'Accuracy', 'Shootings', 'Dribbling', 'Sprint Speed', 'Shot Power',
          'Jumping', 'Penalties']


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = len(set(list1 + list2))
    return float(intersection) / union


def monge_elkan_similarity(s1, s2):
    scores = []
    for token1 in s1:
        token_scores = []
        for token2 in s2:
            token_score = fuzz.token_sort_ratio(token1, token2) / 100.0
            token_scores.append(token_score)
        max_score = max(token_scores)
        scores.append(max_score)
    return sum(scores) / len(scores)


def semantic_matcher(mediated_text, dataset_text):
    # Create TF-IDF vectors for the preprocessed text data
    vectorizer = TfidfVectorizer()
    mediated_vector = vectorizer.fit_transform(mediated_text)
    dataset_vector = vectorizer.transform(dataset_text)

    # Calculate cosine similarity between the vectors
    similarity_score = cosine_similarity(mediated_vector, dataset_vector)[0][0]

    return similarity_score


def calculate_cosine_similarity(word1, word2):
    try:
        vector1 = model[word1.lower()]
        vector2 = model[word2.lower()]
        return model.cosine_similarities(vector1, [vector2])[0]
    except KeyError:
        return 0.0


def calculate_semantic_similarity(attribute1, attribute2):
    stop_words = set(stopwords.words('english'))
    tokens1 = [w.lower() for w in word_tokenize(attribute1) if w.lower() not in stop_words]
    tokens2 = [w.lower() for w in word_tokenize(attribute2) if w.lower() not in stop_words]

    similarity_scores = []
    for token1 in tokens1:
        for token2 in tokens2:
            similarity_scores.append(calculate_cosine_similarity(token1, token2))

    if len(similarity_scores) > 0:
        return max(similarity_scores)
    else:
        return 0.0


num_mediated = len(mediated_scheme)
num_fifa21 = len(fifa21)
word_1 = np.zeros((num_mediated, num_fifa21))
word_2 = np.zeros((num_mediated, num_fifa21))
word_3 = np.zeros((num_mediated, num_fifa21))

for i, attribute1 in enumerate(mediated_scheme):
    for j, attribute2 in enumerate(nba):
        similarity = calculate_semantic_similarity(attribute1, attribute2)
        word_1[i, j] = similarity
        # print(f"Similarity between '{attribute1}' and '{attribute2}': {similarity}")

for i, attribute1 in enumerate(mediated_scheme):
    for j, attribute2 in enumerate(cycling):
        similarity = calculate_semantic_similarity(attribute1, attribute2)
        word_2[i, j] = similarity

for i, attribute1 in enumerate(mediated_scheme):
    for j, attribute2 in enumerate(fifa21):
        similarity = calculate_semantic_similarity(attribute1, attribute2)
        word_3[i, j] = similarity

# print(word_1)
# print(word_2)
# print(word_3)


similarity_matrix1_1 = []  # Jaccard
similarity_matrix2_1 = []
similarity_matrix3_1 = []

similarity_matrix1_2 = []  # elkan
similarity_matrix2_2 = []
similarity_matrix3_2 = []

similarity_matrix1_3 = []  # don`t use
similarity_matrix2_3 = []
similarity_matrix3_3 = []

similarity_matrix1_4 = []  # Word2Vec
similarity_matrix2_4 = []
similarity_matrix3_4 = []

dataset_text_1 = list(map(str, nba))
dataset_text_2 = list(map(str, cycling))
dataset_text_3 = list(map(str, fifa21))

for col1 in mediated_scheme:
    similarities = []
    similarities_2 = []
    similarities_3 = []
    for col2 in nba:
        similarity = jaccard_similarity(col1, col2)
        similarity2 = monge_elkan_similarity(col1, col2)
        similarity3 = semantic_matcher([col1], [col2])  # Pass each text as a list
        similarities.append(similarity)
        similarities_2.append(similarity2)
        similarities_3.append(similarity)
    similarity_matrix1_1.append(similarities)
    similarity_matrix1_2.append(similarities_2)
    similarity_matrix1_3.append(similarities_3)

# Create a matrix as a pandas DataFrame
similarity_matrix1_1 = pd.DataFrame(similarity_matrix1_1, index=mediated_scheme, columns=nba)
similarity_matrix1_2 = pd.DataFrame(similarity_matrix1_2, index=mediated_scheme, columns=nba)
similarity_matrix1_3 = pd.DataFrame(similarity_matrix1_3, index=mediated_scheme, columns=nba)

for col1 in mediated_scheme:
    similarities = []
    similarities_2 = []
    similarities_3 = []
    for col2 in cycling:
        similarity = jaccard_similarity(col1, col2)
        similarity2 = monge_elkan_similarity(col1, col2)
        similarity3 = semantic_matcher([col1], [col2])  # Pass each text as a list
        similarities.append(similarity)
        similarities_2.append(similarity2)
        similarities_3.append(similarity)
    similarity_matrix2_1.append(similarities)
    similarity_matrix2_2.append(similarities_2)
    similarity_matrix2_3.append(similarities_3)

similarity_matrix2_1 = pd.DataFrame(similarity_matrix2_1, index=mediated_scheme, columns=cycling)
similarity_matrix2_2 = pd.DataFrame(similarity_matrix2_2, index=mediated_scheme, columns=cycling)
similarity_matrix2_3 = pd.DataFrame(similarity_matrix2_3, index=mediated_scheme, columns=cycling)

for col1 in mediated_scheme:
    similarities = []
    similarities_2 = []
    similarities_3 = []
    for col2 in fifa21:
        similarity = jaccard_similarity(col1, col2)
        similarity2 = monge_elkan_similarity(col1, col2)
        similarity3 = semantic_matcher([col1], [col2])  # Pass each text as a list
        similarities.append(similarity)
        similarities_2.append(similarity2)
        similarities_3.append(similarity)
    similarity_matrix3_1.append(similarities)
    similarity_matrix3_2.append(similarities_2)
    similarity_matrix3_3.append(similarities_3)

similarity_matrix3_1 = pd.DataFrame(similarity_matrix3_1, index=mediated_scheme, columns=fifa21)
similarity_matrix3_2 = pd.DataFrame(similarity_matrix3_2, index=mediated_scheme, columns=fifa21)
similarity_matrix3_3 = pd.DataFrame(similarity_matrix3_3, index=mediated_scheme, columns=fifa21)

# print(similarity_matrix1_3)
# print(similarity_matrix2_3)
# print(similarity_matrix3_3)


similarity_matrix1_3 = similarity_matrix1_3.to_numpy()

heatmap = plt.imshow(similarity_matrix1_3, cmap='hot', interpolation='nearest')

sentences = []
sentences.append(' '.join(mediated_scheme))
sentences.append(' '.join(fifa21))
sentences = [mediated_scheme, nba, cycling, fifa21]

# print(sentences)

model1 = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model2 = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model3 = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

i = 0
while i < len(mediated_scheme):
    similar_words = model1.wv.most_similar(mediated_scheme[i])
    # print(similar_words[1])
    i = i + 1

# similar_words = model.wv.most_similar("Team")
# print(similar_words)


for word1 in mediated_scheme:
    row_fifa21 = []
    row_cycling = []
    row_nba = []

    for word2 in nba:
        if word1 in model.wv.key_to_index and word2 in model.wv.key_to_index:
            row_nba.append(model.wv.similarity(word1, word2))
        else:
            row_nba.append(0)

    for word2 in cycling:
        if word1 in model.wv.key_to_index and word2 in model.wv.key_to_index:
            row_cycling.append(model.wv.similarity(word1, word2))
        else:
            row_cycling.append(0)

    for word2 in fifa21:
        if word1 in model.wv.key_to_index and word2 in model.wv.key_to_index:
            row_fifa21.append(model.wv.similarity(word1, word2))
        else:
            row_fifa21.append(0)

    similarity_matrix1_4.append(row_nba)
    similarity_matrix2_4.append(row_cycling)
    similarity_matrix3_4.append(row_fifa21)

similarity_matrix1_4 = pd.DataFrame(similarity_matrix1_4, columns=nba, index=mediated_scheme)
similarity_matrix2_4 = pd.DataFrame(similarity_matrix2_4, columns=cycling, index=mediated_scheme)
similarity_matrix3_4 = pd.DataFrame(similarity_matrix3_4, columns=fifa21, index=mediated_scheme)

# numeric_columns = similarity_matrix1_1.select_dtypes(include=[np.number]).columns


# max_df = pd.DataFrame(np.maximum(similarity_matrix1_1[numeric_columns].values,
#                                 similarity_matrix1_2[numeric_columns].values,
#                                 similarity_matrix1_4[numeric_columns].values),
#                      columns=numeric_columns,
#                      index=similarity_matrix1_1.index)
# print(max_df)

weights = [0.4, 0.3, 0.3]

combiner1_1 = DataFrameCombiner(similarity_matrix1_1, similarity_matrix1_2, similarity_matrix1_4)
combiner2_1 = DataFrameCombiner(similarity_matrix2_1, similarity_matrix2_2, similarity_matrix2_4)
combiner3_1 = DataFrameCombiner(similarity_matrix3_1, similarity_matrix3_2, similarity_matrix3_4)

###

max_df1 = combiner1_1.get_max_dataframe()
max_df2 = combiner2_1.get_max_dataframe()
max_df3 = combiner3_1.get_max_dataframe()

###

combiner1_2 = DataFrameCombiner_Average(similarity_matrix1_1, similarity_matrix1_2, similarity_matrix1_4)
combiner2_2 = DataFrameCombiner_Average(similarity_matrix2_1, similarity_matrix2_2, similarity_matrix2_4)
combiner3_2 = DataFrameCombiner_Average(similarity_matrix3_1, similarity_matrix3_2, similarity_matrix3_4)

###

average_df1 = combiner1_2.get_average_dataframe()
average_df2 = combiner2_2.get_average_dataframe()
average_df3 = combiner3_2.get_average_dataframe()

###

combiner1_3 = DataFrameCombiner_MA(similarity_matrix1_1, similarity_matrix1_2, similarity_matrix1_4, weights)
combiner2_3 = DataFrameCombiner_MA(similarity_matrix2_1, similarity_matrix2_2, similarity_matrix2_4, weights)
combiner3_3 = DataFrameCombiner_MA(similarity_matrix3_1, similarity_matrix3_2, similarity_matrix3_4, weights)

###

weighted_average_df1 = combiner1_3.get_weighted_average_dataframe()
weighted_average_df2 = combiner2_3.get_weighted_average_dataframe()
weighted_average_df3 = combiner3_3.get_weighted_average_dataframe()

###

#print(max_df1)
# print(average_df1)
# print(weighted_average_df1)

selector_df1_1 = apply_selector(max_df1)
selector_df2_1 = apply_selector(max_df2)
selector_df3_1 = apply_selector(max_df3)
selector_df1_2 = apply_selector(average_df1)
selector_df2_2 = apply_selector(average_df2)
selector_df3_2 = apply_selector(average_df3)
selector_df1_3 = apply_selector(weighted_average_df1)
selector_df2_3 = apply_selector(weighted_average_df2)
selector_df3_3 = apply_selector(weighted_average_df3)

# (selector_df1_3)

print(selector_df1_1)
#print(selector_df2_1)
#print(selector_df3_1)


data1 = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
         ]

data2 = [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         ]

data3 = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

#print(data1)
#print(data2)
#print(data3)
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)

print(df1)

precision1_1, recall1_1, f1_score1_1 = calculate_f1(selector_df1_1, df1)
precision2_1, recall2_1, f1_score2_1 = calculate_f1(selector_df2_1, df2)
precision3_1, recall3_1, f1_score3_1 = calculate_f1(selector_df3_1, df3)
precision1_2, recall1_2, f1_score1_2 = calculate_f1(selector_df1_2, df1)
precision2_2, recall2_2, f1_score2_2 = calculate_f1(selector_df2_2, df2)
precision3_2, recall3_2, f1_score3_2 = calculate_f1(selector_df3_2, df3)
precision1_3, recall1_3, f1_score1_3 = calculate_f1(selector_df1_3, df1)
precision2_3, recall2_3, f1_score2_3 = calculate_f1(selector_df2_3, df2)
precision3_3, recall3_3, f1_score3_3 = calculate_f1(selector_df3_3, df3)





print("precision1_1:     ", round(precision1_1, 2), "recall1_1:    ", round(recall1_1, 2), "f1_score1_1:     ", round(f1_score1_1, 2))
print("precision2_1:     ", round(precision2_1, 2), "recall2_1:    ", round(recall2_1, 2), "f1_score2_1:     ", round(f1_score2_1, 2))
print("precision3_1:     ", round(precision3_1, 2), "recall3_1:    ", round(recall3_1, 2), "f1_score3_1:     ", round(f1_score3_1, 2))
print("precision1_2:     ", round(precision1_2, 2), "recall1_2:    ", round(recall1_2, 2), "f1_score1_2:     ", round(f1_score1_2, 2))
print("precision2_2:     ", round(precision2_2, 2), "recall2_2:    ", round(recall2_2, 2), "f1_score2_2:     ", round(f1_score2_2, 2))
print("precision3_2:     ", round(precision3_2, 2), "recall3_2:    ", round(recall3_2, 2), "f1_score3_2:     ", round(f1_score3_2, 2))
print("precision1_3:     ", round(precision1_3, 2), "recall1_3:    ", round(recall1_3, 2), "f1_score1_3:     ", round(f1_score1_3, 2))
print("precision2_3:     ", round(precision2_3, 2), "recall2_3:    ", round(recall2_3, 2), "f1_score2_3:     ", round(f1_score2_3, 2))
print("precision3_3:     ", round(precision3_3, 2), "recall3_3:    ", round(recall3_3, 2), "f1_score3_3:     ", round(f1_score3_3, 2))



#return precision, recall, f1

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = similarity_matrix1_1.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_NBA_Jaccard')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(similarity_matrix1_1)), similarity_matrix1_1)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_NBA_Jaccard')
plt.tight_layout()
#plt.show()

####

similarity_matrix = similarity_matrix1_2.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_NBA_Mongo_Elcan')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(similarity_matrix1_2)), similarity_matrix1_2)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_NBA_Mongo_Elcan')
plt.tight_layout()
#plt.show()

####

similarity_matrix = similarity_matrix1_4.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_NBA_Word2Vec')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(similarity_matrix1_4)), similarity_matrix1_4)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_NBA_Word2Vec')
plt.tight_layout()
#plt.show()

####

similarity_matrix = similarity_matrix2_1.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_Cycling_Jaccard')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_Cycling_Jaccard')
plt.tight_layout()
#plt.show()

####

similarity_matrix = similarity_matrix2_2.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_Cycling_Mongo_Elcan')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_Cycling_Mongo_Elcan')
plt.tight_layout()
#plt.show()

####

similarity_matrix = similarity_matrix2_4.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_Cycling_Word2Vec')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_Cycling_Word2Vec')
plt.tight_layout()
#plt.show()

###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = similarity_matrix3_1.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_Fifa21_Jaccard')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_Fifa21_Jaccard')
plt.tight_layout()
#plt.show()

###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = similarity_matrix3_2.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_Fifa21_Jaccard')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_Fifa21_Jaccard')
plt.tight_layout()
#plt.show()

###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = similarity_matrix3_4.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='Similarity_Fifa21_Word2Vec')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('Similarity_Fifa21_Word2Vec')
plt.tight_layout()
#plt.show()

###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = max_df1.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='max_df1')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('max_df1')
plt.tight_layout()
#plt.show()

###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = max_df2.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='max_df2')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('max_df2')
plt.tight_layout()
#plt.show()


###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = max_df3.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='max_df3')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('max_df3')
plt.tight_layout()
#plt.show()


###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = average_df1.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='average_df1')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('average_df1')
plt.tight_layout()
#plt.show()


###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = average_df2.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='average_df2')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('average_df2')
plt.tight_layout()
#plt.show()


###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = average_df3.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='average_df3')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('average_df3')
plt.tight_layout()
#plt.show()


###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = weighted_average_df1.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='weighted_average_df1')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('weighted_average_df1')
plt.tight_layout()
#plt.show()


###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = weighted_average_df2.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='weighted_average_df2')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('weighted_average_df2')
plt.tight_layout()
#plt.show()


###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = weighted_average_df3.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='weighted_average_df3')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('weighted_average_df3')
plt.tight_layout()
#plt.show()



###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = selector_df1_1.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='selector_df1_1')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('selector_df1_1')
plt.tight_layout()
#plt.show()



###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = selector_df1_2.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='selector_df1_2')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('selector_df1_2')
plt.tight_layout()
#plt.show()



###

fig_width = 8  # Width of the plot
fig_height = 6

similarity_matrix = selector_df1_3.to_numpy()

plt.figure(figsize=(8, 6))
heatmap = plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')

# Add text annotations to each square
for i in range(len(mediated_scheme)):
    for j in range(len(mediated_scheme)):
        text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha='center', va='center', color='b')

plt.colorbar(heatmap, label='selector_df1_3')
plt.xticks(range(len(mediated_scheme)), mediated_scheme, rotation=90)
plt.yticks(range(len(mediated_scheme)), mediated_scheme)
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.title('selector_df1_3')
plt.tight_layout()
#plt.show()

