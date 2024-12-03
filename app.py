import streamlit as st
import traceback
from groq import Groq
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone

def initialize_recommendation_system():
    try:
        # Initialize Groq
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Pinecone
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        
        # Get the index
        index_name = "imdb-index"
        index = pc.Index(index_name)
        
        # Check index stats
        index_stats = index.describe_index_stats()
        
        # Initialize vector store
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
            namespace=""
        )
        
        # Initialize LLM
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            api_key=st.secrets["GROQ_API_KEY"],
            temperature=0
        )
        
        # Define prompt template
        template = """You are a movie recommender system that helps users find movies that match their preferences.
        Use the following pieces of context to answer the question at the end.
        For each question, suggest three movies, with a short description of the plot and the reason why the user might like it.
        Format your response in a clear, easy-to-read way with line breaks between movies.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Your response:"""
        
        PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"Error initializing the recommendation system: {str(e)}")
        st.error(traceback.format_exc())
        return None

def get_recommendations(query, qa_chain):
    try:
        with st.spinner('🎬 Finding perfect movies for you...'):
            st.write(f"Searching for query: {query}")
            result = qa_chain.invoke({"query": query})
            recommendations = result['result']
            return recommendations
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        st.error(traceback.format_exc())
        return None

def main():
    # Custom CSS to reduce margins
    st.markdown("""
        <style>
            .block-container {
                padding-left: 2rem !important;
                padding-right: 2rem !important;
                max-width: 95rem !important;
            }
            .stButton button {
                width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state keys if they don't exist
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False

    # Header
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Find your next favorite movie!")

    # Initialize the system if not already done
    if not st.session_state.initialized:
        with st.spinner('Initializing recommendation system...'):
            qa_chain = initialize_recommendation_system()
            if qa_chain:
                st.session_state.qa_chain = qa_chain
                st.session_state.initialized = True

    # Create columns for layout with adjusted ratios
    col1, col2 = st.columns([3, 1])  # Changed ratio from [2, 1] to [3, 1] for better space utilization

    with col1:
        # Search input
        query = st.text_input(
            "What kind of movie are you looking for?",
            placeholder="e.g., 'A sci-fi movie with time travel' or 'A romantic comedy set in New York'",
            key="movie_query"
        )

        # Search button
        if st.button("Get Recommendations 🔍", type="primary"):
            if query:
                recommendations = get_recommendations(query, st.session_state.qa_chain)
                if recommendations:
                    # Process and extract movie details
                    recommendations_list = recommendations.strip().split('\n')
                    formatted_recommendations = []
                    for line in recommendations_list:
                        # Ensure movie names are detected and formatted
                        if "Movie:" in line or line.startswith("*"):
                            formatted_recommendations.append(f"**{line.strip()}**")
                        else:
                            formatted_recommendations.append(line.strip())

                    # Combine into a single formatted block
                    final_output = "\n\n".join(formatted_recommendations)

                    # Display recommendations in one box
                    st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);">
                            <h4>🎥 Movie Recommendations:</h4>
                            <p style="white-space: pre-line;">{final_output}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations found. Please try a different query.")
            else:
                st.warning("Please enter what kind of movie you're looking for!")

if __name__ == "__main__":
    main()