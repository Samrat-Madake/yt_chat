from langchain_community.document_loaders import YoutubeLoader

def load_transcript(youtube_url: str):
    """Loads the transcript of a given YouTube video URL."""
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"Error loading transcript: {e}")
        raise e