from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import validators
import os
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.set_page_config("Summarize from Url",page_icon="🐦")
st.title("Summarize from YT or website Urls")


llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
prompt_text = """
Summarize the text to 300 words
text:{text}
Summarize:
"""
prompt = PromptTemplate(input_variables=['text'],template=prompt_text)
url = st.text_input(placeholder="Enter a Youtube or Website Url",label="Enter a Youtube or Website Url")
if st.button("Generate a Summary"):
    if not url.strip():
        st.error("Please provide a valide Url")
    elif not validators.url(url):
        st.error("Please enter a valid Url")
    else:
        if "youtube.com" in url:
            loader = YoutubeLoader.from_youtube_url(url)
        else:
            loader = UnstructuredURLLoader(urls=[url])
    try: 
        docs = loader.load()
        chain = load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)
        summary = chain.run(docs)
        st.success(summary)
    except Exception as e:
        st.exception(f"Exception:{e}")

