#### this function calculates accuracy of predictions from a confusion matrix
calcAcc <- function(confMatrix) {
  accuracy <- sum(diag(confMatrix)) / sum(confMatrix)
  return(accuracy)
}



#### this function extracts additional text features and adds new columns onto df
addTextFeaturesToDF <- function(df, textBodyVar) {
  ## count the number of exclamation points
  df$excCnt <- str_count(df[[textBodyVar]], '!')
  
  ## count the number of capitalized characters
  df$capCnt <- str_count(df[[textBodyVar]], '[A-Z]')
  
  ## count the number of characters
  df$charCnt <- nchar(df[[textBodyVar]])
  
  ## count the number of question marks
  df$quesCnt <- str_count(df[[textBodyVar]], '\\?')
  
  ## count the number of periods
  df$perCnt <- str_count(df[[textBodyVar]], '\\.')
  
  ## count the number of @ symbols
  df$atCnt <- str_count(df[[textBodyVar]], '@')
  
  ## count the number of commas
  df$comCnt <- str_count(df[[textBodyVar]], ',')
  
  ## count the number of line breaks
  # df$lbCnt <- str_count(df[[textBodyVar]], '\n')  # regular expression doesn't work
  
  ## contains curse words
  curseWords <- c('shit', 'fuck', 'damn', 'bitch', 'crap', 'piss', 
                  'dick', 'cock', 'pussy', 'whor', 'fool', 'suck',
                  'ass', 'fag', 'gay', 'bastard', 'slut', 'skank', 'douche', 
                  'moron', 'stupid', 'ignorant', 'dumb', 'cunt', 'shit')
  curseRegexPat <- paste(curseWords, collapse='|')
  df$cursCnt <- str_count(df[[textBodyVar]], curseRegexPat)  
  
  ## return
  return(df)
}



#### this function extracts additional date/time features and adds new columns onto df
addDateTimeFeaturesToDF <- function(df, datetimeVar) {
  ## extract day of the week 
  df$day <- weekdays(df[[datetimeVar]], abb=TRUE)
  df$day[is.na(df$day)] <- 'unknown'
  
  ## extract day type (weekend vs. weekday)
  df$dayType <- ifelse(df$day=='unknown', 'unknown',
                          ifelse(df$day %in% c('Sat', 'Sun'), 'weekend', 'weekday'))
  
  ## extract time of the day
  df$hour <- as.integer(format(df[[datetimeVar]], '%H'))
  df$tod <- ifelse(is.na(df$hour), 'unknown',
                      ifelse(df$hour >= 6 & df$hour < 12, 'morning', 
                             ifelse(df$hour >= 12 & df$hour < 18, 'afternoon',
                                    ifelse(df$hour >= 18 & df$hour < 24, 'evening', 'dawn'))))
  
  ## return data
  return(df)
}



#### this function marks comments that were posted during typical work day hours 
#### (i.e. from 8 a.m. to 5 p.m. on Monday through Friday)
markPostsNormWorkHours <- function(df, datetimeVar=NULL, dayVar=NULL, hourVar=NULL) {
  
  ## if a function call hasn't specified day and hour variables
  if (is.null(dayVar) | is.null(hourVar)) {
    
    ## and if day and hour variables cannot be found in df
    if (!all(c('day', 'hour') %in% colnames(df))) {
      
      ## (re)create these fields 
      df <- addDateTimeFeaturesToDF(df, datetimeVar)      
    }
    
    ## specify day and hour variables
    dayVar <- 'day'
    hourVar <- 'hour'
  }
  
  ## conditional for posts generated on weekdays
  wkdayCond <- !(df$day %in% c('Sat', 'Sun')) 
  
  ## conditional for posts generated between 8 a.m. to 5 p.m.
  btwn8amTo5pmCond <- df$hour >= 8 & df$hour <= 17
  
  ## conditional for normal workday hours
  normWkdayHrsCond <- wkdayCond & btwn8amTo5pmCond
  
  ## logical column for "posted on normal work hours"
  df$PNWH <- normWkdayHrsCond
  
  ## return
  return(df)
}



#### this function marks comments that were posted within 6 hours prior to normal work day hours
markPosts6HrsPriorNormWorkHours <- function(df, tsVar, dayVar=NULL) {
  next  
}


# weekdays <- subset(train, dayType=='weekday')
# weekends <- subset(train, dayType=='weekend')
# 
# table(weekdays$insult, weekdays$tod)
# round(prop.table(table(weekdays$insult, weekdays$tod), 2), 2)
# table(weekends$insult, weekends$tod)
# round(prop.table(table(weekends$insult, weekends$tod), 2), 2)
# 
# table(weekdays$insult, weekdays$hour)
# round(prop.table(table(weekdays$insult, weekdays$hour), 2), 2)
# table(weekends$insult, weekends$hour)
# round(prop.table(table(weekends$insult, weekends$hour), 2), 2)




#### this function creates sparse word-frequency matrix from a vector of text bodies
createSparseWordFreqMtrx <- function(textBodyVector, minWordAppPerc) {
  ## creating a corpus
  corpus <- Corpus(VectorSource(textBodyVector))

  ## converting texts to lowercase
  # http://stackoverflow.com/questions/24771165/r-project-no-applicable-method-for-meta-applied-to-an-object-of-class-charact
  corpus <- tm_map(corpus, content_transformer(tolower))
  
  ## remove punctuations
  corpus <- tm_map(corpus, removePunctuation)
  
  ## remove stop words
  corpus <- tm_map(corpus, removeWords, stopwords('english'))
  
  ## stem words
  corpus <- tm_map(corpus, stemDocument)
  
  ## create a document-term matrix
  dtm <- DocumentTermMatrix(corpus)

  ## remove spare terms
  spdtm <- removeSparseTerms(dtm, 1-minWordAppPerc)  # select terms that occur at least 2% of the times

  ## create (sparse) word-frequency matrix
  swfdf <- as.data.frame(as.matrix(spdtm))
  
  ## preface column with "word_"
  if (ncol(swfdf) > 0) {
    colnames(swfdf) <- paste0('word_', colnames(swfdf))    
  }

  ## return
  return(swfdf)
}



#### this function replaces the  sparse word-frequen
replaceTextBodyWithSparseWordFreqMtrx <- function(df, textBodyVar, minWordAppPerc) {
  ## create sparse word-frequency matrix
  sparseWordFreqMtrx <- createSparseWordFreqMtrx(df[[textBodyVar]], minWordAppPerc)
    
  ## join the sparse word-frequency matrix with the original df
  df <- cbind(df, sparseWordFreqMtrx)
  
  ## remove the original text body column
  df[[textBodyVar]] <- NULL
  
  ## return 
  return(df)
}



#### this function splits a dataset into train and cross-validation datasets
splitToTrainAndCV <- function(df, outcomeVar, randNumSeed=123, splitRatio=0.75) {
  set.seed(randNumSeed)
  split <- sample.split(df[[outcomeVar]], SplitRatio=splitRatio)
  trainDF <- df[split==TRUE, ]
  cvDF <- df[split==FALSE, ]
  return(list(train=trainDF, cv=cvDF))
}




#### this function creates 
