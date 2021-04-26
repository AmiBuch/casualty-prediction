

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pytesseract
import cv2
from skimage import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string 
import flask
def hand_core(im):
	text = pytesseract.image_to_string(im)
	return text

def get_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
	return cv2.medianBlur(image, 5)

def thresholding(image):
	return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
def word_tokenizer(sents):
	tokenized_glop = nltk.tokenize.word_tokenize(sents)
	return tokenized_glop    
stopwords_en = stopwords.words("english")
app = flask.Flask(__name__, template_folder = 'templates')
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))
	if flask.request.method == 'POST':
		answer = str(flask.request.form['cor_ans'])
        	marks = int(flask.request.form['marks_me'])
        	Image_dumb = flask.request.form['dumb']
        	img = io.imread(Image_dumb)
        	img = get_grayscale(img)
        	img = thresholding(img)
        	img = remove_noise(img)
        	yo_ans = hand_core(img)
        	plog = word_tokenizer(yo_ans)
        	plog_real = word_tokenizer(answer)
        	raw_deal = [w.lower() for w in plog if w not in stopwords_en]
        	real_deal = [w.lower() for w in plog if w not in stopwords_en]
        	deal1 = ""
        	for i in raw_deal:
			deal1 += str(i)
			deal1 += " "          
        	deal2 = ""
        	for i in real_deal:
			deal2 += str(i)
			deal2 += " "       
          
        	documents = (deal1, deal2)
        	tfidf_vectorizer = TfidfVectorizer()
        	tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        	my_array = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
        	relief = my_array[:][-1][-1]
        	if 0.15 < relief < 0.25:
			prediction = str("marks scored : 2")
          
		elif 0 < relief < 0.15:
			prediction = str("marks scored : 1")
	
        	else:
			prediction = str("marks scored : 3")
          
       
        	return flask.render_template('main.html', result = prediction,)
        
	
    
    
        
        

if __name__ == '__main__':
    app.run()



