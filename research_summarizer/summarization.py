from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import torch


def sliding_window_summarization(
    text: str,
    tokenizer: AutoTokenizer,
    pipe: pipeline,
    max_length: int = 512,
    overlap: int = 128,
    redundancy_threshold: float = 0.8,
    summary_length: int = 100,  # Desired summary length
) -> str:
    """
    Summarizes a long text using a sliding window approach with overlap and
    redundancy reduction.

    Args:
        text: The input text to summarize.
        tokenizer: The tokenizer for the summarization model.
        pipe: The summarization pipeline.
        max_length: The maximum sequence length for the model.
        overlap: The overlap between consecutive chunks.
        redundancy_threshold: The similarity threshold for redundancy reduction.
        summary_length: The desired length of the summary.

    Returns:
        A summarized version of the input text.
    """
    try:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception as e:
        print(f"Tokenization error: {e}")
        return ""

    summaries = []
    previous_summary_embedding = None

    # Initialize sentence transformer model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Calculate chunk size based on max_length and overlap
    chunk_size = max_length - overlap

    while tokens:  # Process all chunks
        # Get the next chunk
        chunk = tokens[:max_length]
        tokens = tokens[chunk_size - overlap :]  # Retain overlap tokens

        # Decode the chunk back into text
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)

        try:
            # Generate summary for the current chunk
            output = pipe(chunk_text)
        except Exception as e:
            print(f"Summarization pipeline error: {e}")
            continue

        # Get the summary text from the pipeline output
        summary = output[0]["summary_text"]

        # Compute sentence embedding for redundancy checking
        summary_embedding = sentence_model.encode(summary)

        # Convert summary_embedding to a PyTorch tensor
        summary_embedding = torch.tensor(summary_embedding).clone().detach()

        # Redundancy reduction using sentence embeddings
        if previous_summary_embedding is not None:
            # Convert previous_summary_embedding to a PyTorch tensor if it is not already
            previous_summary_embedding = (
                torch.tensor(previous_summary_embedding).clone().detach()
            )

            # Normalize embeddings before calculating cosine similarity
            similarity = util.cos_sim(
                summary_embedding / summary_embedding.norm(),
                previous_summary_embedding / previous_summary_embedding.norm(),
            )

            if similarity > redundancy_threshold:
                print(
                    f"Skipping redundant summary (similarity {similarity.item():.3f})"
                )
                continue  # Skip highly similar summaries

        # Append the non-redundant summary
        summaries.append(summary)
        previous_summary_embedding = summary_embedding

    # Join all summaries together
    full_summary = " ".join(summaries)

    # Truncate at the last sentence boundary before summary_length
    if len(full_summary) > summary_length:
        last_period = full_summary.rfind(".", 0, summary_length)
        if last_period != -1:
            full_summary = full_summary[: last_period + 1]
        else:  # If no period found, truncate at summary_length
            full_summary = full_summary[:summary_length]

    return full_summary
