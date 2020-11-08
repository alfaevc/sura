#load("test2.RData")
#load('testing.RData')
#load('MLas.RDAta')
load('models.RData')
library(knitr)
library(tidyverse)
library(tidytext)
library(stringr)
library(caret)      # model evaluation
library(tm)
library(glmnet)
library(text2vec)     # NLP tools
library(xgboost)      # XGBoost implementation
library(textstem)     # word lemmatization
library(randomForest)
library(e1071)
library(neuralnet)
library(naivebayes)
library(qdap)
library(dplyr)
library(tm)
library(wordcloud)
library(plotrix)
library(dendextend)
library(ggplot2)
library(ggthemes)
library(RWeka)


getTextDF = function(textFile) { # read in text file uploaded by user (PUBMED Syntax)
  file = readLines(textFile)
  
  file1 = file[file != '']
  sub_ex = substr(file1, start = 1, stop = 4)
  unique(sub_ex)
  labels = unique(sub_ex)
  
  # initialize values for looping
  mat = matrix(data = rep(''), ncol = length(labels))
  colnames(mat) = labels
  
  j = 1                # row number
  i = 2                # skip first line
  prev_lab = "PMID"    # first label is always 'PMID'
  
  while(i <= length(file)) {
    label = substr(file[i], start = 1, stop = 4) # first 4 characters contains label
    if(label == "    ") { # case where line continues
      mat[j, prev_lab] = paste(mat[j, prev_lab], substr(file[i], start = 7, stop = nchar(file[i])), sep = ' ')
    } else if(label == "" | i == length(file)) { # case where we move onto the next study
      mat = rbind(mat, rep('', length(labels)))
      j = j + 1
    } else if(mat[j, label] == '') { # case where there is no info already in space
      mat[j, label] = substr(file[i], start = 7, stop = nchar(file[i]))
    } else { # case where there is already info already in space
      mat[j, label] = paste(mat[j, label], substr(file[i], start = 7, stop = nchar(file[i])), sep = ' ##o0## ')
    }
    if(label != "    ") { # change previous label if line was not continued
      prev_lab = label
    }
    i = i + 1
  }
  
  # construct data frame
  df = data.frame(mat)
  
  colnames(df)[names(df)=="AB.."] <- "abstract"
  colnames(df)[names(df)=="TI.."] <- "title"
  df = df %>%
    mutate(title = as.character(title)) %>%
    mutate(abstract = as.character(abstract))
  return(df)
}

clean.texts <- function(text) {
  text = removeNumbers(text)
  text = tolower(text)
  text = bracketX(text)
  text = removePunctuation(text)
  text = stripWhitespace(text)
  text = removeWords(text, stopwords('en'))
  return(text)
}

fact2num <- function(x) {
  return (as.numeric(as.character(x)))
}

wordFreq <- function(text, all.df) {
  words = strsplit(text, "\\W")[[1]]
  words = words[words != ""] 
  words = unique(words)
  df = all.df[0,]
  df[1,] = rep(0)
  words[words == "function"] <- "funct"
  words[words == "next"] <- "nex"
  for (i in 1:length(words)) {
    if (words[i] %in% names(df)) {
      df[1, words[i]] = 1
    }
  }
  return (df) 
}

bigramFreq <- function(text, all.df) {
  temp.df = data.frame(abstract=text)
  temp.df = temp.df %>%
    unnest_tokens(bigram, "abstract", token = "ngrams", n = 2)
  bigrams = unique(temp.df$bigram)
  bigrams = bigrams[bigrams != ""] 
  df = all.df[0,]
  df[1,] = rep(0)
  bigrams <- sub(' ', '.', bigrams)
  for (i in 1:length(bigrams)) {
    if (bigrams[i] %in% names(df)) {
      df[1, bigrams[i]] = 1
    }
  }
  return (df) 
}


SURA.predict <- function(all.df, model.name, input, trained.model=NULL, title=TRUE) {
  cleaned.input = clean.texts(input)
  if (title) {
    test = wordFreq(cleaned.input, all.df)
  } else {
    test = bigramFreq(cleaned.input, all.df)
  }
  
  model = trained.model
  if (model.name == "DF") {
    if (is.null(trained.model)) {
      model = randomForest(L ~ ., data = all.df)
    }
    test.pred = fact2num(predict(model, newdata = test))
  } else if (model.name == "SVM") {
    all.df[,1] <- fact2num(all.df[,1])
    test[,1] <- fact2num(test[,1])
    if (is.null(trained.model)) {
      model = svm(L ~ ., data = all.df, kernel = "linear", cost = 10, scale = FALSE)
    }
    test.pred = predict(model, newdata = test, type = "raw")
  } else if (model.name == "NB") {
    all.df[,1] <- fact2num(all.df[,1])
    test[,1] <- fact2num(test[,1])
    if (is.null(trained.model)) {
      model = naiveBayes(L ~ ., data = all.df, laplace = 3)
    }
    test.pred = predict(model, newdata = test, type = "raw")
  } else if (model.name == "NN") {
    for (i in 1:ncol(all.df)) {
      all.df[,i] <- fact2num(all.df[,i])
    }
    for (i in 1:ncol(test)) {
      test[,i] <- fact2num(test[,i])
    }
    if (is.null(trained.model)) {
      hidden_layers = c(400, 100, 100)
      ns = colnames(all.df)
      s = paste("L ~", paste(ns[c(-1)], collapse = " + "))
      nn.fun <- as.formula(s)
      model = neuralnet(nn.fun, data=all.df, hidden=hidden_layers, act.fct = "logistic", linear.output = FALSE)
    }
    test.pred = compute(model, test)$net.result
    
  } else { #lasso
    for (i in 1:ncol(all.df)) {
      all.df[,i] <- fact2num(all.df[,i])
    }
    for (i in 1:ncol(test)) {
      test[,i] <- fact2num(test[,i])
    }
    if (is.null(trained.model)) {
      set.seed(10)
      x = model.matrix(L ~. , all.df)[,-1]
      #lambda_seq <- 10^seq(2, -2, by = -.1)
      #cv.lasso <- cv.glmnet(x, all.df$L, alpha = 1, lambda = lambda_seq, nfolds = 5, family = "binomial")
      
      #lam = cv.lasso$lambda.min
      model <- glmnet(x, all.df$L, alpha = 1, family = "binomial")
    }
    n = dim(test)[1]
    test[n+1,] = rep(0)
    x.test = model.matrix(L~. , test)[,-1]
    test.pred = predict(model, newx = x.test)
  }
  return (test.pred)
}


outputView = function(x, view) {
  y = x
  if(view == "S") {
    y = ifelse(x < .5, "Non-Trial", "Trial")
  }
  y
}

getDF = function(inpType, ti, ab, file, view) {
  
  df = NULL
  # input$file1 will be NULL initially. After the user selects
  # and uploads a file, it will be a data frame with 'name',
  # 'size', 'type', and 'datapath' columns. The 'datapath'
  # column will contain the local filenames where the data can
  # be found.
  
  if(inpType == 'Title') {
    svm = outputView(SURA.predict(title.bwords.df, "SVM", ti, title.svm, title=TRUE), view)
    if(view == "D") {
      decFor = outputView(SURA.predict(title.bwords.df, "DF", ti, title.dforest, title=TRUE), view)
      nb = outputView(SURA.predict(title.bwords.df, "NB", ti, title.nb, title=TRUE)[2], view)
      nn = outputView(SURA.predict(title.bwords.df, "NN", ti, title.nn, title=TRUE), view)
      df = as.data.frame(t(c(svm, nb, decFor, nn)))
      names(df) = c('SVM', 'NB', 'DF', 'NN')
    }
    else {
      df = as.data.frame(t(svm))
      names(df) = c('Guess')
    }
  } else if(inpType == 'Abstract') { # need to change to accomodate different models
    svm = outputView(SURA.predict(ab.bwords.df, "SVM", ab, ab.svm, title=FALSE), view)
    if(view == 'D') {
      decFor = outputView(SURA.predict(ab.bwords.df, "DF", ab, ab.dforest, title=FALSE), view)
      nb = outputView(SURA.predict(ab.bwords.df, "NB", ab, ab.nb, title=FALSE)[2], view)
      nn = outputView(SURA.predict(ab.bwords.df, "NN", ab, ab.nn, title=FALSE)[2], view)
      df = data.frame(t(c(svm, nb, decFor, nn)))
      names(df) = c('SVM', 'NB', 'DF', 'NN')
    }
    else {
      df = as.data.frame(t(svm))
      names(df) = c('Guess')
    }
  }
  else if(inpType == 'Text File') {
    if (is.null(file)) return(NULL)
    textDF = getTextDF(file$datapath)
    if (view == 'D') {
      df = textDF %>%
        select(PMID, title) %>%
        mutate(svm = 0) %>%
        mutate(nn = 0) %>%
        mutate(nb = 0) %>%
        mutate(df = 0)
      for(i in 1:length(textDF$abstract)) {
        BoW = textDF[i, 'title']
        df[i, 'svm'] = outputView(SURA.predict(ab.bwords.df, "SVM", ab, ab.svm, title=FALSE), view)
        df[i, 'nn'] = outputView(SURA.predict(ab.bwords.df, "NN", ab, ab.nn, title=FALSE), view)
        df[i, 'nb'] = outputView(SURA.predict(ab.bwords.df, "NB", ab, ab.nb, title=FALSE)[2], view)
        df[i, 'df'] = outputView(SURA.predict(ab.bwords.df, "DF", ab, ab.dforest, title=FALSE), view)
      }
    }
    else {
      df = textDF %>%
        select(PMID, title) %>%
        filter(!is.na(PMID)) %>%
        mutate(Guess = 0)
      for(i in 1:length(textDF$abstract)) {
        BoW = textDF[i, 'title']
        df[i, 'Guess'] = outputView(SURA.predict(ab.bwords.df, "SVM", ab, ab.svm, title=FALSE), view)
      }
    }
  }
  df
}


shinyServer({
  function(input, output) {
    a = eventReactive(input$predictButton, {getDF(input$inputType, input$Title, input$Abstract, input$file1, input$view)})
    output$contents = renderTable({a()})
  }
})