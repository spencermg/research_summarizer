from openai import OpenAI
import google.generativeai as genai
import anthropic
from transformers import pipeline
from transformers import AutoTokenizer, pipeline


class llm_summarizer:
    def __init__(self, openai_key, anthropic_key, gemini_key):
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.gemini_key = gemini_key
        genai.configure(api_key = gemini_key)
        self.openai = OpenAI(api_key = openai_key)
        self.claude = anthropic.Anthropic(api_key = anthropic_key)

    def summarize_bart(self, article_text, summaries, device):
        """
        Summarize article using Bart pretrained transformer

        Args:
            article_text (str): The full text of the article to be summarized.
            summaries (dict): Summaries for each LLM model.
            device (int): 0 if using GPUs, otherwise -1.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        # Summarize the article using the BART model
        model_name = "facebook/bart-large-cnn"

        # Call the sliding window summarization function
        summaries["Bart"] = summarize_text(
            model_name,
            device,
            article_text,
            512,
        )

        return summaries

    def summarize_falcons(self, article_text, summaries, device):
        """
        Summarize article using Falcons pretrained transformer

        Args:
            article_text (str): The full text of the article to be summarized.
            summaries (dict): Summaries for each LLM model.
            device (int): 0 if using GPUs, otherwise -1.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        # Summarize the article using the Falconsai/medical_summarization model
        model_name = "Falconsai/medical_summarization"

        # Call the sliding window summarization function
        summaries["Falconsai"] = summarize_text(
            model_name,
            device,
            article_text,
            512,
        )

        return summaries

    def summarize_bigbird(self, article_text, summaries, device):
        """
        Summarize article using BigBird pretrained transformer

        Args:
            article_text (str): The full text of the article to be summarized.
            summaries (dict): Summaries for each LLM model.
            device (int): 0 if using GPUs, otherwise -1.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """


        # Summarize the article using the "google/bigbird-pegasus-large-pubmed" model
        model_name = "google/bigbird-pegasus-large-pubmed"

        # Call the sliding window summarization function
        summaries["BigBird"] = summarize_text(
            model_name,
            device,
            article_text,
            3968,
        )

        return summaries

    def summarize_gpt(self, article_text, system_message, summaries):
        """
        Sends an article to the OpenAI API to generate a summary.

        Args:
            article_text (str): The full text of the article to be summarized.
            system_message (str): Prompt to indicate what the LLM should do.
            summaries (dict): Summaries for each LLM model.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        if self.openai_key != "":
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": article_text},
            ]
            response = self.openai.chat.completions.create(
                model = "gpt-4o-mini", 
                messages = messages, 
                stream = False
            )
            summary = response.choices[0].message.content
            summaries["OpenAI"] = summary
            return summaries
        else:
            return summaries

    def summarize_claude(self, article_text, system_message, summaries):
        """
        Sends an article to the Anthropic API to generate a summary.

        Args:
            article_text (str): The full text of the article to be summarized.
            system_message (str): Prompt to indicate what the LLM should do.
            summaries (dict): Summaries for each LLM model.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        if self.anthropic_key != "":
            try:
                response = self.claude.messages.create(
                    model = "claude-3-haiku-20240307",
                    max_tokens = 1000,
                    system = system_message,
                    messages = [{"role": "user", "content": article_text}],
                )
                summary = response.content[0].text
                summaries["Anthropic"] = summary
                return summaries
            except Exception as e:
                raise Exception(f"Error summarizing article: {e}")
        else:
            return summaries
        
    def gemini(self, article_text, system_message, summaries):
        """
        Sends an article to the Gemini API to generate a summary.

        Args:
            article_text (str): The full text of the article to be summarized.
            system_message (str): Prompt to indicate what the LLM should do.
            summaries (dict): Summaries for each LLM model.

        :return: summaries *(str)*: \n
            Summaries from each model.
        """

        if self.gemini_key != "":
            gemini = genai.GenerativeModel(
                model_name = 'gemini-1.5-flash',
                system_instruction = system_message,
            )
            response = gemini.generate_content(article_text)
            summary = response.candidates[0].content.parts[0].text
            summaries["Gemini"] = summary
            return summaries
        else:
            return summaries


def summarize_text(model_name, device, article_text, max_length, overlap=128, summary_length=250):
    """
    Prepare pretrained transformer model for summarization and pass through sliding window summarizer.

    Args:
        model_name (str): Name of transformer model being used
        device (int): 0 if using GPUs, otherwise -1
        article_text (str): Full text from an article
        max_length (int): The maximum sequence length for the model
        overlap (int, optional): The overlap between consecutive chunks (Default: 128)
        summary_length (int, optional): The desired length of the summary (Default: 250)

    :return: summary *(str)*: \n
        Summarization of the full-text article
    """

    # Iitialize the tokenizer and pipeline for summarization
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarization_pipeline = pipeline(
        "summarization", 
        model = model_name, 
        device = device,
    )

    # Generate summary
    summary = sliding_window_summarization(
        text = article_text,
        tokenizer = tokenizer,
        pipe = summarization_pipeline,
        max_length = max_length,
        overlap = overlap,
        summary_length = summary_length,
    )

    return summary


def sliding_window_summarization(
    text: str,
    tokenizer: AutoTokenizer,
    pipe: pipeline,
    max_length: int = 512,
    overlap: int = 128,
    summary_length: int = 900,
) -> str:
    """
    Summarizes a long text using a sliding window approach with overlap and
    redundancy reduction.

    Args:
        text (str): The input text to summarize.
        tokenizer (AutoTokenizer): The tokenizer for the summarization model.
        pipe (pipeline): The summarization pipeline.
        max_length (int): The maximum sequence length for the model.
        overlap (int): The overlap between consecutive chunks.
        summary_length (int): The desired length of the summary.

    :return: full_summary *(str)*: \n
        A summarized version of the input text
    """

    # No need to summarize if the text is already shorter than the summary length
    if len(text.split()) <= summary_length:
        return text
    
    # Tokenize input text
    try:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception as e:
        print(f"Tokenization error: {e}")
        return ""
    
    # No need to use sliding windows if the model can process the entire input text
    if len(tokens) <= max_length:
        chunk_text = tokenizer.decode(tokens, skip_special_tokens=True)
        output = pipe(chunk_text)
        summary = output[0]["summary_text"]
        return summary

    # Sumarize input text in chunks using sliding window
    summaries = []
    while len(tokens) > 0:
        # Decode the chunk back into text for summarization
        chunk_text = tokenizer.decode(tokens[:max_length], skip_special_tokens=True)

        # For the last chunk, add more padding such that it will contain max_length tokens
        if len(tokens) == max_length:
            tokens = []
        elif len(tokens) < (2 * max_length - overlap):
            tokens = tokens[-max_length:]
        else:
            tokens = tokens[(max_length - overlap):]

        # Summarize current chunk
        try:
            output = pipe(chunk_text)
            summaries.append(output[0]["summary_text"])
        except Exception as e:
            print(f"Summarization pipeline error: {e}")

    return sliding_window_summarization(
        " ".join(summaries),
        tokenizer,
        pipe,
        max_length,
        0,
        summary_length,
    )
