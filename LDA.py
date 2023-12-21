import os
import pdfplumber
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy
import spacy

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

names = []


def getfiles(dir):
    texts = []
    for filename in os.listdir(dir):
        if filename.endswith('.pdf'):
            print(filename)
            names.append(filename)
            pdf_path = os.path.join(dir, filename)
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                texts.append(text)
    return texts


text_content = getfiles('Articles')

dict = {'names': names, 'descriptions': text_content}

# importing stopwords
gist_file = open("gist_stopwords.txt", "r")
try:
    content = gist_file.read()
    stops = content.split(",")
finally:
    gist_file.close()


names = dict["names"]
data = dict["descriptions"]

# This function is lemmatizing the words in the text to their root form using spaCy3.0 model en_core_web_sm
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)


# the data is converted to its root words
lemmatized_texts = lemmatization(data)

# removal of punctuations and symbols
data_words = gen_words(lemmatized_texts)


# converting the data words to a dictionary of words so that now each word in the list data_word has its ID
id2word = corpora.Dictionary(data_words)


# vectorizing the data by mapping the index to the frequency of the word
corpus = []
for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)

# Training the LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=30,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")

# Obtaining the topics from the LDA model
topics = lda_model.print_topics(num_topics=30, num_words=30)

print("Topics: \n")
# Print the topics and their top words
for topic_id, topic in topics:
    print(f"Topic {topic_id + 1}: {topic}")

# Get the document-topic matrix
document_topic_matrix = lda_model.get_document_topics(corpus)

# Convert the document-topic matrix to a NumPy array for further analysis if needed
document_topic_matrix_np = gensim.matutils.corpus2dense(document_topic_matrix, num_terms=lda_model.num_topics).T

# Print the shape of the document-topic matrix
print("Shape of Document-Topic Matrix:", document_topic_matrix_np.shape)

# Print the content of the document_topic_matrix
for i, doc_topics in enumerate(document_topic_matrix):
    print(names[i])
    for topic_id, topic_prob in doc_topics:
        print(f"  Topic {topic_id + 1}: Probability {topic_prob:.6f}")
    print()



# Set the size of the heatmap
plt.figure(figsize=(15, 10))

# Create a heatmap
sns.heatmap(document_topic_matrix_np, cmap="YlGnBu", annot=True, fmt=".2f", cbar_kws={'label': 'Topic Probability'})

# Set axis labels and title
plt.xlabel('Topics')
plt.ylabel('Documents')
plt.title('Document-Topic Matrix')

# Show the plot
plt.show()

topics = lda_model.show_topics(formatted=False)
for topic_id, words in topics:
    word_probs = np.array(words)
    words, probs = zip(*word_probs)
    plt.figure(figsize=(20, 10))
    plt.barh(words, probs, color='skyblue')
    plt.xlabel('Word Probability')
    plt.title(f'Topic {topic_id + 1} Word Distribution')
    plt.show()
