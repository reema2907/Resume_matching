import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(job_skills, resume_skills):
    """
    Calculate the cosine similarity between job skills and resume skills.
    
    Parameters:
        job_skills (list): List of skills mentioned in the job description.
        resume_skills (list): List of skills listed in the resume.
    
    Returns:
        float: Cosine similarity score between job skills and resume skills.
    """
    # Convert lists of skills to strings
    job_skills_str = ", ".join(job_skills)
    resume_skills_str = ", ".join(resume_skills)
    
    # Tokenize job skills and resume skills
    text_list = [job_skills_str, resume_skills_str]
    
    # Vectorize the text
    vectorizer = CountVectorizer().fit_transform(text_list)
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(vectors)
    
    # Return the cosine similarity score
    return similarity_score[0, 1]
