library(feather)
library(text2vec)
library(dplyr)
library(anytime)
library(ggplot2)
library(stringr)

### read data in, RT Nieuws archives
RTLN = read_feather("/home/hadoop/datatransfer/RTL_Nieuws_archive.feather")

RTLN = RTLN %>% filter(value !="")

### text mine, tokenize etc.
RTLNEWS_tokens = RTLN$value %>%
  word_tokenizer

##### and use the tokens to create an iterator and vocabular
it_train = itoken(
  RTLNEWS_tokens, 
  ids = RTLN$nid,
  progressbar = TRUE
)

stopwoorden = readRDS("/home/hadoop/datatransfer/ds.RDS")

vocab = create_vocabulary(
  it_train, 
  ngram = c(ngram_min = 1L, ngram_max = 1L),
  stopwords = stopwoorden
)

pruned_vocab = prune_vocabulary(
  vocab, 
  term_count_min = 50 ,
  doc_proportion_max = 0.75
  #,doc_proportion_min = 0.001
)

print("*** vocab generated****")
print(pruned_vocab)

vectorizer <- vocab_vectorizer(
  pruned_vocab, 
  # don't vectorize input
  grow_dtm = FALSE, 
  # use window of 5 for context words
  skip_grams_window = 5L
)

tcm <- create_tcm(it_train, vectorizer)
dim(tcm)

#######  Glove word embeddings

glove = GlobalVectors$new(word_vectors_size = 200, vocabulary = pruned_vocab, x_max = 10)
glove$fit(tcm, n_iter = 20)
word_vectors = glove$get_word_vectors()

dim(word_vectors)
word_vectors[1,]

WV <- word_vectors["parijs", , drop = FALSE] 
cos_sim = sim2(x = word_vectors, y = WV, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 20)




