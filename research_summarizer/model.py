import anthropic
import google.generativeai as genai
import ollama
import pandas as pd
from datasets import Dataset, DatasetDict
from openai import OpenAI
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from tqdm import tqdm

client = ollama.Client()
client.pull("llama3.2")
client.pull("gemma2")
client.pull("phi3:medium-128k")

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

    def summarize_llm(self, article_text, system_message, summaries, model_name):
        """
        Sends an article to the OpenAI API to generate a summary.

        Args:
            article_text (str): The full text of the article to be summarized
            system_message (str): Prompt to indicate what the LLM should do
            summaries (dict): Summaries for each LLM model
            model_name (str): Name of LLM model being used

        :return: summaries *(str)*: \n
            Summaries from each model
        """

        model_dict = {
            "Llama":"llama3.2",
            "Gemma":"gemma2",
            "Phi3":"phi3:medium-128k",
        }

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": article_text},
        ]
        summary = ollama.chat(model=model_dict[model_name], messages=messages)['message']['content']
        summaries[model_name] = summary
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
    summary = sliding_window_summarization(article_text, tokenizer, summarization_pipeline, max_length, overlap, summary_length)
    return summary


def sliding_window_summarization(text, tokenizer, pipe, max_length, overlap, summary_length):
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


def fine_tune_transformer(model_name, df_articles, device, results_dir, overlap=128, summary_length=250):
    """
    Fine tune one of the pretrained transformer models

    Args:
        model_name_transformer (str): Name of the best-performing transformer model
        df_articles (pandas.DataFrame): Dataframe containing PMCIDs and corresponding full-text articles
        device (int): 0 if using GPUs, otherwise -1
        results_dir (pathlib.Path): Path to directory where outputs are saved
        overlap (int, optional): The overlap between consecutive chunks.
        summary_length (int, optional): The desired length of the summary.

    :return: df_summaries (pandas.DataFrame): \n
        Mapping from PMCIDs to summaries from the new fine-tuned model
    """

    model_dict = {
        "Bart":"facebook/bart-large-cnn", 
        "Falconsai":"Falconsai/medical_summarization", 
        "BigBird":"google/bigbird-pegasus-large-pubmed",
    }
    model_name = model_dict[model_name]

    print("Generating summaries for fine-tuned transformer model...")
    train_df, test_df = train_test_split(df_articles, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    tokenized_articles = dataset.map(lambda batch: tokenize_batch(batch, tokenizer), batched=True, batch_size=None)

    args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy='epoch',
        save_strategy='epoch',
        weight_decay=0.01,
        learning_rate=2e-5,
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        train_dataset=tokenized_articles['train'],
        eval_dataset=tokenized_articles['test'],
    )
    trainer.train()
    model.save_pretrained(results_dir / "tuned_model.json")
    model = AutoModelForSeq2SeqLM.from_pretrained(results_dir / "tuned_model.json").to(device)
    summarization_pipeline = pipeline("summarization", model=model, device=device)

    summaries = {}
    for _, df_article in tqdm(df_articles.iterrows(), total=len(df_articles), desc="Summarizing articles after fine-tuning"):
        summaries[df_article['pmcid']] = [sliding_window_summarization(
            text = df_article["full_text"],
            tokenizer = tokenizer,
            pipe = summarization_pipeline,
            max_length = 512,
            overlap = overlap,
            summary_length = summary_length,
        )]
    
    df_summaries = pd.DataFrame(summaries)
    print("Article summaries generated for fine-tuned transformer model!")
    return df_summaries


def tokenize_batch(batch, tokenizer, max_length=512, padding="max_length", return_tensors="pt"):
    """
    Tokenizes the input and target text from the batch using the specified tokenizer.

    Args:
        batch (dict): A dictionary containing the input and target text with keys for full text and abstract
        tokenizer (transformers.AutoTokenizer): Model being used to tokenize text
        max_length (int, optional): The maximum length of the tokenized sequences (Default: 512)
        padding (str, optional): Padding strategy (Default: "max_length")
        return_tensors (str, optional): The type of tensors to return (Default: "pt")

    :return: encoding *(dict)*: \n
        The tokenized and encoded representations of the input and target texts
    """

    return tokenizer(
        batch["full_text"],
        text_target=batch["abstract"],
        max_length=max_length,
        truncation=True,
        padding=padding,
        return_tensors=return_tensors,
    )
