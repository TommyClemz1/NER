import streamlit as st 
from time import sleep
from stqdm import stqdm # for getting animation after submit event 
import pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit



def draw_all(
    key,
    plot=False,
):
    st.write(
        """
        # NER Web App
        
        This Named Entity Recognition Based Web App that can identify and classify text data.👩🏾‍💻🔥
        
        This App is built using pretrained transformers which are capable of doing wonders with the Textual data.
        
        ```
         Named Entity Recognition
        ```
        """
    )

    

with st.sidebar:
    draw_all("sidebar")



def main():
    st.title("NER Web App")
    menu = ["Named Entity Recognition General",
            "Named Entity Recognition Domain"]
    choice = st.sidebar.selectbox("Choose Which model you want to use", menu)


    if choice=="Named Entity Recognition General":

        st.write("""
                 
                 
Named Entity Recognition (NER) is a subfield of NLP and Artificial Intelligence (AI) that identifies and classifies named entities (such as person names, locations, organizations) in text, helping machines understand the context just like humans.
        """)


        nlp = spacy.load("en_core_web_trf")
        st.subheader("Text Based Named Entity Recognition")
        raw_text = st.text_area("Your Text","Enter Text Here")
        if st.button("Analyze Text"):
            if raw_text !="Enter Text Here":
                doc = nlp(raw_text)
                for _ in stqdm(range(50), desc="Please wait a bit. The model is     fetching the results !!"):
                    sleep(0.1)
                spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels,   title= "List of Entities")













    elif choice=="Named Entity Recognition Domain":
        
        st.write("""
                 
                This is a Natural Language Processing (NLP) Based Web App that specializes in Named Entity Recognition (NER) for a specific domain.

        """)
        
        st.write("""
               
Natural Language Processing (NLP) is a computational   technique to understand human language in the way it is   spoken and written.

        """)
        
        st.write("""
                 
                 
Named Entity Recognition (NER) is a subfield of NLP and Artificial Intelligence (AI) that identifies and classifies named entities (such as person names, locations, organizations) in text, helping machines understand the context just like humans.
        """)
        
        
        st.image('qr_img.png')



if __name__ == '__main__':
     main()