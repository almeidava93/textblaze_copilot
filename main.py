import pandas as pd
import numpy as np
import streamlit as st
import openai
from openai.embeddings_utils import distances_from_embeddings
from functools import lru_cache

openai.api_key = st.secrets['openai']['key']

st.experimental_memo
def load_embeddings():
    df=pd.read_hdf('embeddings.hdf', key="embeddings")
    df['embeddings'] = df['embeddings'].apply(np.array)
    return df

df = load_embeddings()

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None,
    use_outside_info=False
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    # If it is to use outside context info or not
    if use_outside_info:
      about_outside_info = ", and use information outside this context if needed"
    else:
      about_outside_info = ", and if the question can't be answered based on the context, say \"I don't know\""

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based mainly on the context below{about_outside_info}\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    
st.title('TextBlaze Q&A')
st.subheader('Powered by OpenAI')

st.text_area('Write your question here about TextBlaze...', key='question')
st.checkbox('Do you want to use information outside TextBlaze documentation?', key='outside-info')
st.number_input('Answer size (number of tokens)', min_value=0, max_value=3000, step=10, value=150, key='n_tokens')

answer = answer_question(df, 
                        question=st.session_state['question'], 
                        use_outside_info=st.session_state['outside-info'], 
                        max_tokens=st.session_state['n_tokens'])

answer = answer.replace('\n', '<br>')
style = """
<style>
#answer {
    background-color: rgb(38, 39, 48);
    color: rgb(250, 250, 250);
    padding: 20px;
    border-radius: 20px;
    font: "sans serif";
}
</style>
"""

if st.session_state['question'] != '': 
    st.markdown(f"<p id='answer'>{answer}</p>{style}", unsafe_allow_html=True)