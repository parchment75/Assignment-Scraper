import tweepy
import credentials
import time
import string
import pandas
import re
from nltk.corpus import stopwords
from datetime import date
stopwords_english = set(stopwords.words('english'))
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

lemmer = WordNetLemmatizer()
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt




def assignment_scraper(user_prompt):
############################################authentication object constructured by feeding the imported OAuth class with API keys - may need more code to obtain intial access token

    try:
        auth = tweepy.OAuthHandler(credentials.CONSUMER_KEY,
                                   credentials.CONSUMER_SECRET)  # create OAuthHandler instance to exchange request token for access token
        auth.set_access_token(credentials.ACCESS_TOKEN, credentials.ACCESS_SECRET)
        api = tweepy.API(auth, wait_on_rate_limit=True)
    except:
        print("Error: Failed to Authenticate")
############################################################################## Scraper starts by scraping based on search terms passed to the function

    Tweets = []
    Tweets_screen_name = []
    Tweets_user_location = []
    Tweets_created = []
    User_created = []
    today = date.today()

    text_query = user_prompt + ' -filter:retweets'  # filters out retweets
    count = 250
    try:
        scrape_tweets = tweepy.Cursor(api.search, q=text_query, lang="en").items(count)  # scrapes tweets to create an iterable object, then iterates over them in a for loop to save the text to a list
        for tweet in scrape_tweets:
            Tweets.append(tweet.text)
            Tweets_screen_name.append(tweet.user.screen_name)
            Tweets_user_location.append(tweet.user.location)
            Tweets_created.append(tweet.created_at)
            User_created.append(tweet.user.created_at)

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)

############################################################################# Pre-process tweets for lexicon based sentiment analysis

    # Happy Emoticons
    emoticons_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'
    ])

    # Sad Emoticons
    emoticons_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('
    ])

    # all emoticons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)

    def clean_tweets(tweet): #function for searching for regex pattern for pre-processing
        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)

        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)

        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)

        # remove@(mentions)
        tweet = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', tweet)

        # remove colon sumbol and remove mentions
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)

        # replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)

        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []

        for word in tweet_tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in emoticons and  # remove emoticons
                    word not in string.punctuation):  # remove punctuation
                # tweets_clean.append(word)
                lem_words = lemmer.lemmatize(word)  # uncomment if want to use lemmatization
                # stem_word = stemmer.stem(word)  # stemming word (process of linguistic normalization which reduces words to their word root word or chops off the derivational affixes e.g. connection becomes connect)
                # tweets_clean.append(stem_word)
                tweets_clean.append(lem_words)

        return tweets_clean

    cleaned_tweets = []
    for tweet in Tweets:
        cleaned_tweets.append((clean_tweets(tweet)))

############################################################################# Lexicon sentiment analysis using TextBlob's sentiment corpora
    sentiment_tweets = []
    for word in cleaned_tweets:
        a = TextBlob(str(word))
        sentiment_tweets.append(str(a.sentiment.polarity))

############################################################################# Use pandas to create a Dataframe and save cleaned tweets to csv
    new_list = list(map(' '.join, cleaned_tweets))
    pd = pandas.DataFrame(
        list(zip(Tweets_screen_name, User_created, Tweets, Tweets_created, Tweets_user_location, sentiment_tweets)),
        columns=['Screen Name', 'User Creation Date', 'Tweet', 'Post date', 'User Location', 'Sentiment'])
    pd.to_csv(str(user_prompt) + ' ' + str(today))

############################################################################# Count the sentiment

    count_total = 0
    count_pos = 0
    count_neg = 0
    count_neut = 0

    Overall_sentiment = [float(i) for i in
                         sentiment_tweets]  # convert overall sentiment to a float so it can be counted

    for i in Overall_sentiment:
        if (i > 0):
            count_pos = count_pos + 1
            count_total = count_total + 1
        # Overall_sentiment.append(1) #do I want these in another list?
        elif (i < 0):
            count_neg = count_neg + 1
            count_total = count_total + 1
        # Overall_sentiment.append(-1)
        else:
            # Overall_sentiment.append(0)
            count_neut += 1

            count_total = count_total + 1

    print("Total tweets with sentiment:", count_total)
    print("positive tweets:", count_pos)
    print("negative tweets:", count_neg)
    print("neutral tweets:", count_neut)

    ############################################################################# Graph breakdown of sentiment

    left = ['Positive tweets', 'Negative tweets', 'Neutral tweets']
    height = [count_pos, count_neg, count_neut]
    tick_label = ['Positive tweets', 'Negative tweets', 'Neutral tweets']
    plt.bar(left, height, tick_label=tick_label, width=0.8, color=['green', 'red', 'blue'])
    plt.xlabel('Sentiment')
    plt.ylabel('Total tweets')
    plt.title('Overall Sentiment')
    plt.title('Overall Sentiment for ' + str(user_prompt))
    plt.savefig(str(user_prompt) + ' ' + str(today) + ' graph')

    ############################################################################# word clouds are so hot right now

    unique_string = (" ").join(
        new_list)  # convert 'new_list' to a string, the word cloud function does not seem to like lists
    wc = WordCloud(width=800, height=800, background_color='white', min_font_size=10)
    wc.generate(unique_string)
    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(user_prompt) + ' ' + str(today) + ' words')


assignment_scraper('#sportsrort')
assignment_scraper('#crimeminister')
assignment_scraper('#auspol')
assignment_scraper('@senbmckenzie')
assignment_scraper('@ScottMorrisonMP')
assignment_scraper('@GregHuntMP')
