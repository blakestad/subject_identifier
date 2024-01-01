from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

from subject_identifier.initialize_model import math_categories, tfidf_title_vectorizer, tfidf_abstract_vectorizer, title_mnb, abstract_mnb




def identify_subjects(input_title=None,input_abstract=None, title_weight=0.4, abstract_weight=0.6, sum_threshold=.25):
    """
    Takes in the title and abstract of a mathematics paper, returns the expected math subject classications for that paper.

    Parameters
    ----------
    input_title : string
        Title of input paper (Default value = None)
    input_abstract : string
        Abstract of the input paper (Default value = None)
    title_weight : numeric
        Relative weighting of the title (Default value = 0.4)
    abstract_weight : numeric
        Relative weighting of the abstract (Default value = 0.6)
    sum_threshold : numeric
        Threshold of weighted sum of probabilities needed to return a 'True' value for a given category name  (Default value = 0.25)

    Returns
    -------
    category_classifications: DataFrame
        The category classifications and corresponding probabilities (rounded to 5 decimal places) from the model.
    """
    #Initialize category classication DataFrame
    category_classifications = math_categories[['category_name']].copy()
    category_classifications['guess']=False
    category_classifications['prob']=0.0
    category_classifications.set_index('category_name',inplace=True)

    if ((input_title == None) or (input_title == '')) and ((input_abstract == None) or (input_abstract == '')):
        raise ValueError('At least one input is required.')

    elif(input_title != None) and ((input_abstract == None) or (input_abstract == '')):
        querry_title_tfidf = tfidf_title_vectorizer.transform([input_title])
        threshold = 0.1

        for key, value in math_categories.iterrows():
            category_name=value['category_name']
            category_classifications.at[category_name,'guess']= (title_mnb[category_name].predict_proba(querry_title_tfidf)[0][1] >= threshold)
            category_classifications.at[category_name,'prob']= round(title_mnb[category_name].predict_proba(querry_title_tfidf)[0][1],5)

    
    elif ((input_title == None) or (input_title == '')) and (input_abstract != None):
        querry_abstract_tfidf = tfidf_abstract_vectorizer.transform([input_abstract])
        threshold = 0.1

        for key, value in math_categories.iterrows():
            category_name=value['category_name']
            category_classifications.at[category_name,'guess']= (abstract_mnb[category_name].predict_proba(querry_abstract_tfidf)[0][1] >= threshold)
            category_classifications.at[category_name,'prob']= round(abstract_mnb[category_name].predict_proba(querry_abstract_tfidf)[0][1],5)


    else:
        querry_title_tfidf = tfidf_title_vectorizer.transform([input_title])
        querry_abstract_tfidf = tfidf_abstract_vectorizer.transform([input_abstract])

        def sum_predict(category_name):
            """
            Takes in a category name, returns the weighted sum of the predicted probabilities from both the title and abstract classifiers, as well as a guess about whether or not the paper is of the given category.

            Parameters
            ----------
            category_name : string
                Name of the mathematical subject category


            Returns
            -------
            prob_sum: numeric
                The weighted sum of the predicted probabilities from the title and abstract classifiers. 
            sum_predict: boolean
                The prediction for whether the paper is of this subjected classication, based on if prob_sum exceeds sum_threshold
            """

            abstract_probability = abstract_mnb[category_name].predict_proba(querry_abstract_tfidf)[0][1]
            title_probability = title_mnb[category_name].predict_proba(querry_title_tfidf)[0][1]
            prob_sum = (abstract_weight*abstract_probability + title_weight*title_probability)
            sum_predict = (prob_sum  >= sum_threshold)

            return prob_sum, sum_predict
        
    
        for key, value in math_categories.iterrows():
            category_name=value['category_name']
            category_classifications.at[category_name,'guess']= sum_predict(category_name)[1]
            category_classifications.at[category_name,'prob']= round(sum_predict(category_name)[0],5)

    category_classifications.reset_index(inplace=True)
    category_classifications_sorted = category_classifications.sort_values(by='prob',ascending=False)
    
    return category_classifications_sorted