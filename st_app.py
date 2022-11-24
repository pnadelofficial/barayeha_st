import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from dateutil.parser import parse
import streamlit as st
from dadmatools.models.normalizer import Normalizer
from dadmatools.embeddings import get_embedding, get_all_embeddings_info, get_embedding_info
from deep_translator import GoogleTranslator
from stqdm import stqdm
stqdm.pandas()

normalizer = Normalizer(
    full_cleaning=False,
    unify_chars=True,
    refine_punc_spacing=True,
    remove_extra_space=True,
    remove_puncs=False,
    remove_html=False,
    remove_stop_word=False,
    replace_email_with="<EMAIL>",
    replace_number_with=None,
    replace_url_with="",
    replace_mobile_number_with=None,
    replace_emoji_with=None,
    replace_home_number_with=None
)

word_embedding = get_embedding('glove-wiki')

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_data():
    for_tweets = pd.read_csv('for_tweets_full.csv')
    for_tweets.date_posted = for_tweets.date_posted.progress_apply(parse)
    for_tweets['text'] = for_tweets.tweet.progress_apply(normalizer.normalize)
    for_tweets['embedding'] = for_tweets.text.progress_apply(word_embedding.embedding_text)
    return for_tweets
for_tweets = get_data()

def cosine_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def search(search_term, for_tweets, entries=5):
    translated = GoogleTranslator(source='auto', target='fa').translate(text=search_term)
    translated_embedding = word_embedding.embedding_text(translated)

    for_tweets['sim_score'] = for_tweets['embedding'].progress_apply(lambda x: cosine_sim(translated_embedding, x))
    df = for_tweets.sort_values('sim_score', ascending=False)[0:entries]

    df['translated'] = df['text'].progress_apply(lambda x: GoogleTranslator(source='auto', target='en').translate(text=x))
    return df

st.markdown('# Barayeha Semantic Search')
st.markdown(
    '<small>Assembled by Peter Nadel</small>'
    ,unsafe_allow_html=True
)
st.markdown('<hr>', unsafe_allow_html=True)

search_term = st.text_input('Search in English.', 'Sexual orientation')
entries = st.number_input('Choose number of excerpts.', min_value=1, value=5)
df = search(search_term, for_tweets, entries=entries)

st.markdown(
    f'<h2>{search_term}</h2>'
    ,unsafe_allow_html=True
)

for i in range(entries):
    st.markdown(
        f'<small>Similarity Score: {round(df.sim_score.to_list()[i], 3)}</small>'
        ,unsafe_allow_html=True
    )
    st.markdown(
        f'<small>Date posted: {df.date_posted.to_list()[i]}</small>'
        ,unsafe_allow_html=True
    )
    st.markdown(
        f'<small>Favorites: {df.favorites.to_list()[i]}, Retweets: {df.retweets.to_list()[i]}</small>'
        ,unsafe_allow_html=True
    )
    st.markdown(
        f'<p>{df.translated.to_list()[i]}</p>'
        ,unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="text-align: right">{df.text.to_list()[i]}</p>'
        ,unsafe_allow_html=True
    )
    st.markdown('<hr>', unsafe_allow_html=True)