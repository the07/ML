from gensim import corpora, models, matutils
from wordcloud  import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from os import path

#Load the data
corpus = corpora.BleiCorpus('./topic_data/ap/ap.dat', './topic_data/ap/vocab.txt')

NUM_TOPICS = 100

#Build the topic model
model = models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=None)

#Iterate over all the topics in the model
for ti in range(model.num_topics):
    words = model.show_topic(ti, 64)
    tf = sum(f for _, f in words)
    with open('topics.txt', 'w') as output:
        output.write('\n'.join('{}:{}'.format(w, int(1000. * f / tf)) for w, f in words))
        output.write("\n\n\n")

topics = matutils.corpus2dense(model[corpus], num_terms=model.num_topics)
weight = topics.sum(1)
max_topic = weight.argmax()

words = model.show_topic(max_topic, 64)
word_dict = {}

for each in words:
    word_dict[each[0]] = each[1]

# This function will actually check for the presence of pytagcloud and is otherwise a no-op
word_Cloud = WordCloud()
wc = word_Cloud.generate_from_frequencies(frequencies=word_dict)

plt.title("Popular Words")
plt.imshow(wc, interpolation="bilinear")
wc.to_file("popular_words.png")
plt.axis("off")
plt.show()

num_topics_used = [len(model[doc]) for doc in corpus]
fig,ax = plt.subplots()
ax.hist(num_topics_used, np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')
fig.tight_layout()
fig.savefig('Figure_04_01.png')


# Now, repeat the same exercise using alpha=1.0
# You can edit the constant below to play around with this parameter
ALPHA = 1.0

model1 = models.ldamodel.LdaModel(
    corpus, num_topics=NUM_TOPICS, id2word=corpus.id2word, alpha=ALPHA)
num_topics_used1 = [len(model1[doc]) for doc in corpus]

fig,ax = plt.subplots()
ax.hist([num_topics_used, num_topics_used1], np.arange(42))
ax.set_ylabel('Nr of documents')
ax.set_xlabel('Nr of topics')

# The coordinates below were fit by trial and error to look good
ax.text(9, 223, r'default alpha')
ax.text(26, 156, 'alpha=1.0')
fig.tight_layout()
fig.savefig('Figure_04_02.png')
