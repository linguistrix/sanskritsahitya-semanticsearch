import streamlit as st
from google import genai
from google.genai import types
from pinecone import Pinecone

client = genai.Client(api_key=st.secrets["gemini_api_key"])
pinecone = Pinecone(api_key=st.secrets["pinecone_api_key"])
index = pinecone.Index("sanskrit-sahitya-org")

@st.cache_data
def query_pinecone(search_query):
    return index.query(
        vector=client.models.embed_content(
            model="gemini-embedding-001",
            contents=[search_query],
            config=types.EmbedContentConfig(task_type="QUESTION_ANSWERING")).embeddings[0].values, 
        top_k=5,
        include_metadata=True,
        include_values=False
    )

st.title("Sanskrit Sahitya Semantic Search")
st.markdown(
    """    
    This note is on a free API tier and thus supports a limited number of requests. It may become temporarily unavailable if too many people use the service.    
    
    ---
    This is a demo app for trying out semantic search over the SanskritSahitya.org corpus. 

    You can find shlokas through free-form search queries about the content of the shloka.

    Currently the following texts have been included in the Search data: Raghuvansham, Kumarasambhavam, Ramayanam, Mahabharatam, Meghadutam, Kiratarjuniyam, Rtusamharam.

    You can mix and match English, Sanskrit or other languages in the query.
    
    **Examples** - Try entering one of these
      - verse about an old king who is like a lamp in the morning
      - cloud is just smoke and vapor
      - विघ्न होते हुए भी जो नहीं डगमगाते
"""
)

if prompt := st.chat_input("Search query (e.g. verse about an old king being like a lamp in the morning)"):
    if not prompt.strip():
        st.error("Please enter some text.")
    else:        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response ... Please wait."):
            response = query_pinecone(prompt)
            if response.matches:
                for match in response.matches:
                    st.markdown(f"[https://sanskritsahitya.com/{match.id}](https://sanskritsahitya.com/{match.id}})")                    
            else:
                st.error("No match found, or API limit exhausted.")
