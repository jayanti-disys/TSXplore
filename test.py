# # Import the training set
import pandas as pd
from bs4 import BeautifulSoup
import re


def get_data ():
    dataset = pd.read_csv(r"C:\Users\janani.balaraman\Downloads\Sentiment Analysis\Train.csv\Train.csv")
    dataset.head(20)
    dataset.tail(20)
    print(dataset.shape)

    # # Exploratery data analysis
    dataset.describe()

    # # Sentiment count
    dataset['label'].value_counts()

    # # Spliting the training dataset
    #split the dataset
    #train dataset
    train_text = dataset.text [:30000]
    train_label = dataset.label [:30000]
    #test dataset
    test_text = dataset.label [30000:]
    test_label = dataset.label [30000:]
    print (train_text.shape,train_label.shape)
    print (test_text.shape,test_label.shape)
    return train_text, test_text


def text_processing():
    """"
    This part can be used ...
    ....
    """
    # # Text normalization
    import nltk
    from nltk.tokenize.toktok import ToktokTokenizer
    #Tokenization of text
    tokenizer=ToktokTokenizer()
    #Setting English stopwords
    stopword_list=nltk.corpus.stopwords.words('english')
    return stopwords


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()
    #Removing the square brackets


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
    #Apply function on review column
    dataset['text'] = dataset['text'].apply(denoise_text)

# # Removing special character
#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern,'',text)
    return text
#Apply function on review column
dataset['text'] = dataset['text'].apply(remove_special_characters)

# # Text stemming
#Stemming the text
def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
dataset['text'] = dataset['text'].apply(simple_stemmer)

# # Removing stopwords
from nltk.corpus import stopwords
#set stopwords to english
stop = set(stopwords.words('english'))
print(stop)
#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text
#Apply function on review column
dataset['text'] = dataset['text'].apply(remove_stopwords)

# # Normalized train reviews
#Normalized train reviews
norm_train_text = dataset.text[:30000]
norm_train_text[0]

# # Normalized text reviews
#Normalized test reviews
norm_test_text = dataset.text[10000:]
norm_test_text[15000]

# # Bags of words model
from sklearn.feature_extraction.text import CountVectorizer
#Count vectorizer for bag of words
cv = CountVectorizer(min_df = 0, max_df = 1, binary = False, ngram_range = (1,3))
#transformed train reviews
cv_train_text = cv.fit_transform(norm_train_text)
#transformed test reviews
cv_test_text = cv.transform(norm_test_text)
print('BOW_cv_train:',cv_train_text.shape)
print('BOW_cv_test:',cv_test_text.shape)
#vocab=cv.get_feature_names()-toget feature names

# # Term frequency - Inverse document frequency model (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
#Tfidf vectorizer
tv = TfidfVectorizer(min_df = 0, max_df = 1, use_idf = True, ngram_range = (1,3))
#transformed train reviews
tv_train_text = tv.fit_transform(norm_train_text)
#transformed test reviews
tv_test_text=tv.transform(norm_test_text)
print('Tfidf_train:',tv_train_text.shape)
print('Tfidf_test:',tv_test_text.shape)

# # Labeling the sentiment text
from sklearn.preprocessing import LabelBinarizer
#labeling the sentient data
lb=LabelBinarizer()
#transformed sentiment data
label_data=lb.fit_transform(dataset['label'])
print(label_data.shape)

# # Split the sentiment data
#Spliting the sentiment data
train_sentiments = label_data[:30000]
test_sentiments = label_data[10000:]
print(train_sentiments)
print(test_sentiments)

# # Modelling the dataset
from sklearn.linear_model import LogisticRegression,SGDClassifier
#training the model
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_text,train_sentiments)
print(lr_bow)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_text,train_sentiments)
print(lr_tfidf)

# # Logistic regression model performance on test dataset
#Predicting the model for bag of words
lr_bow_predict = lr.predict(cv_train_text)
print(lr_bow_predict)
#Predicting the model for tfidf features
lr_tfidf_predict = lr.predict(tv_train_text)
print(lr_tfidf_predict)

# # Accuracy of the model
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#Accuracy score for bag of words
lr_bow_score = accuracy_score (test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score = accuracy_score (test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)

# # Print the classification report
#Classification report for bag of words
lr_bow_report=classification_report(test_sentiments,lr_bow_predict,target_names=['Positive','Negative'])
print(lr_bow_report)
#Classification report for tfidf features
lr_tfidf_report=classification_report(test_sentiments,lr_tfidf_predict,target_names=['Positive','Negative'])
print(lr_tfidf_report)

# # Confusion matrix
#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,lr_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,lr_tfidf_predict,labels=[1,0])
print(cm_tfidf)

# # Stochastic gradient descent or Linear support vector machines for bag of words and tfidf features
#training the linear svm
svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)
#fitting the svm for bag of words
svm_bow=svm.fit(cv_train_text,train_label)
print(svm_bow)
#fitting the svm for tfidf features
svm_tfidf=svm.fit(tv_train_text,train_label)
print(svm_tfidf)

# # Model performance on test data
#Predicting the model for bag of words
svm_bow_predict=svm.predict(cv_test_text)
print(svm_bow_predict)
#Predicting the model for tfidf features
svm_tfidf_predict=svm.predict(tv_test_text)
print(svm_tfidf_predict)

# # Accuracy of the model
#Accuracy score for bag of words
svm_bow_score=accuracy_score(test_sentiments,svm_bow_predict)
print("svm_bow_score :",svm_bow_score)
#Accuracy score for tfidf features
svm_tfidf_score=accuracy_score(test_sentiments,svm_tfidf_predict)
print("svm_tfidf_score :",svm_tfidf_score)

# # Print the classification report
#Classification report for bag of words
svm_bow_report=classification_report(test_sentiments,svm_bow_predict,target_names=['Positive','Negative'])
print(svm_bow_report)
#Classification report for tfidf features
svm_tfidf_report=classification_report(test_sentiments,svm_tfidf_predict,target_names=['Positive','Negative'])
print(svm_tfidf_report)

# # Plot the confusion matrix
#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,svm_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,svm_tfidf_predict,labels=[1,0])
print(cm_tfidf)

# # Multinational Naive bayes for bag of words and tfidf features
from sklearn.naive_bayes import MultinomialNB
#training the model
mnb=MultinomialNB()
#fitting the svm for bag of words
mnb_bow=mnb.fit(cv_train_text,train_label)
print(mnb_bow)
#fitting the svm for tfidf features
mnb_tfidf=mnb.fit(tv_train_text,train_label)
print(mnb_tfidf)

# # Model performance on test data
#Predicting the model for bag of words
mnb_bow_predict=mnb.predict(cv_test_text)
print(mnb_bow_predict)
#Predicting the model for tfidf features
mnb_tfidf_predict=mnb.predict(tv_test_text)
print(mnb_tfidf_predict)

# # Accuracy of the model
#Accuracy score for bag of words
mnb_bow_score=accuracy_score(test_sentiments,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)
#Accuracy score for tfidf features
mnb_tfidf_score=accuracy_score(test_sentiments,mnb_tfidf_predict)
print("mnb_tfidf_score :",mnb_tfidf_score)

# # Print the classification report
#Classification report for bag of words
mnb_bow_report=classification_report(test_sentiments,mnb_bow_predict,target_names=['Positive','Negative'])
print(mnb_bow_report)
#Classification report for tfidf features
mnb_tfidf_report=classification_report(test_sentiments,mnb_tfidf_predict,target_names=['Positive','Negative'])
print(mnb_tfidf_report)

# # Plot the confusion matrix
#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,mnb_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,mnb_tfidf_predict,labels=[1,0])
print(cm_tfidf)

# # Positive and Negative words by using WordCloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS

# # Word Cloud for positive words
#word cloud for positive review words
plt.figure(figsize=(10,10))
positive_text=norm_train_text[1]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plt.imshow(positive_words,interpolation='bilinear')
plt.show

# # Word Cloud for negative words
#Word cloud for negative review words
plt.figure(figsize=(10,10))
negative_text=norm_train_text[8]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
negative_words=WC.generate(negative_text)
plt.imshow(negative_words,interpolation='bilinear')
plt.show
