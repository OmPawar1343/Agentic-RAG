from sentence_transformers import SentenceTransformer

def get_embedding_model():
    """
    Returns MiniLM embedding model
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    return model
