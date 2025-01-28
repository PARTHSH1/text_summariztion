import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Set page configuration
st.set_page_config(page_title="Summarize the listed URL", page_icon="🦜")
st.title("🦜 Langchain: Summarize Text from YT or Website")
st.subheader('Summarize URL')

# Sidebar for API key input
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API key", type='password')

# Main interface for URL input
generalize_url = st.text_input("Enter the URL of YouTube or website to summarize")

# Initialize LLM
if groq_api_key:
    llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")

# Prompt template
prompt_template = """
Provide a summary of the following content:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarize button
if st.button("Summarize the website or YouTube transcript"):
    if not groq_api_key.strip() or not generalize_url.strip():
        st.error("Please enter the required information.")
    elif not validators.url(generalize_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading and summarizing content..."):
                # Determine the loader based on the URL
                if "youtu" in generalize_url:
                    loader = YoutubeLoader.from_youtube_url(youtube_url=generalize_url, add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls=[generalize_url], ssl_verify=True)
                
                # Load data
                data = loader.load()

                # Summarize the content
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.invoke(data)
                
                # Display the summary
                st.success("Summary:")
                st.write(output_summary)
        except Exception as e:
            st.error(f"An error occurred: {e}")