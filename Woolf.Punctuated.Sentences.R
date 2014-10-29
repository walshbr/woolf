#loads text
text.v <- scan("./Mrs.Dalloway.txt", what="character", sep="\n")
text.v
#cleans texts of blanks
not.blanks.v <- which(text.v!=" ")
clean.text.v <- text.v[not.blanks.v]
#puts text into one long chunk
novel.v <- paste(clean.text.v, collapse=" ")
#lower cases text
novel.lower.v <- tolower(novel.v)
#splits text into characters
woolf.split.l <- strsplit(novel.lower.v, "")
str(woolf.split.l)
woolf.text <- unlist(woolf.split.l)
#stores index positions of all open quotes and smart quotes
open.smart.quotes <- which(woolf.text=="“")
close.smart.quotes <- which(woolf.text=="”")
#grabs number of punctuated phrases in novel.
phrase.number <- length(open.smart.quotes)
punctuated.phrases <- list()
punctuated.indexes <- list()
#loop to store those phrases in a vector 'punctuated.phrases' for processing
for(i in 1:phrase.number){
  phrase.start <- open.smart.quotes[i]
  phrase.end <- close.smart.quotes[i]
  #pulls out all the necessary characters for sentence
  phrase.characters <- woolf.text[phrase.start:phrase.end]
  #stores the index numbers for ALL characters within the quotation marks. not just the opening and closing index numbers.
  punctuated.indexes <- c(punctuated.indexes, phrase.start:phrase.end)
  punctuated.phrases[i] <- paste(phrase.characters, sep="", collapse="")
}
punctuated.indexes <- unlist(punctuated.indexes)
#now we celebrate. "punctuated sentences" contains all those phrases enclosed by smart quotes
#punctuated.indexes contains the index numbers of all characters occurring within smart quotes
#preps a dispersion plot of the punctuated characters.
woolf.time <- seq(1:length(woolf.text))
punct.count <- rep(NA, length(woolf.time))
punct.count[punctuated.indexes] <- 1
plot(punct.count, main="Dispersion Plot of Punctuated Dialogue in Mrs. Dalloway",
     xlab="Novel Time", ylab="Speech in quotation marks", type="h", ylim=c(0,1), yaxt='n')
#produces a sorted listing of the word frequencies in the punctuated phrases.
words.punctuated.phrases <- paste(punctuated.phrases, collapse=" ")
words.punctuated.phrases <- strsplit(words.punctuated.phrases, "\\W")
str(words.punctuated.phrases)
words.punctuated.phrases <- unlist(words.punctuated.phrases)
not.blanks <- which(words.punctuated.phrases!="")
words.punctuated.phrases <- words.punctuated.phrases[not.blanks]
words.punctuated.phrases.freqs <- table(words.punctuated.phrases)
sorted.words.punctuated.phrases.freqs <- sort(words.punctuated.phrases.freqs, decreasing=TRUE)
