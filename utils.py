import wikipedia
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

def get_wikipedia_content(topic):
    """Fetch Wikipedia page content for a given topic."""
    try:
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.PageError:
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Ambiguous topic. Please be more specific. Options: {e.options}")
        return None

def split_text(text, chunk_size=256, chunk_overlap=20):
    """Splits text into overlapping chunks."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokenizer.convert_tokens_to_string(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - chunk_overlap
    return chunks
