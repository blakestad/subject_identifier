import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def initialize():
    """
    Initializes the Tfidf-Vectorizers used for the recommend module.



    Parameters
    ----------


    Returns
    -------
    math_categories: DataFrame
        Contains the arXiv ids, category names, and descriptions of the mathematical categories used on the arXiv. Has columns ['arxiv_id', 'category_name', 'description']

    tfidf_title_vectorizer: TfidfVectorizer
        A tfidf-vectorizer for math paper titles.

    tfidf_abstract_vectorizer: TfidfVectorizer
        A tfidf-vectorizer for math paper abstracts.

    title_mnb: dictionary
        A dictionary of MultinomialNB classifiers. The keys are the category names from math_categories, the values are MultinomialNB classifiers for titles with respect to the assocaited category.

    abstract_mnb: dictionary
        A dictionary of MultinomialNB classifiers. The keys are the category names from math_categories, the values are MultinomialNB classifiers for abstracts with respect to the assocaited category.
    """

    # Load the categories
    math_categories = pd.read_csv('subject_identifier/categories/math_categories.csv',index_col=0)    

    # load the vectorizer
    tfidf_title_vectorizer = joblib.load('subject_identifier/arxiv_data/tfidf_title_vectorizer.pkl')
    tfidf_abstract_vectorizer = joblib.load('subject_identifier/arxiv_data/tfidf_abstract_vectorizer.pkl')

    #Load the MultinomialNB classifiers    
    title_mnb = joblib.load('subject_identifier/arxiv_data/title_mnb_dict.pkl')
    abstract_mnb = joblib.load('subject_identifier/arxiv_data/abstract_mnb_dict.pkl')

    return math_categories, tfidf_title_vectorizer, tfidf_abstract_vectorizer, title_mnb, abstract_mnb


math_categories, tfidf_title_vectorizer, tfidf_abstract_vectorizer, title_mnb, abstract_mnb = initialize()
