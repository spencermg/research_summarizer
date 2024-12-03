import argparse
import gzip
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import pytz
import requests
import research_summarizer.utils as utils
import research_summarizer.model as model
import seaborn as sns
import time as time_module
import torch
import warnings
from datetime import datetime, timedelta, time
from io import BytesIO
from pathlib import Path
from sentence_transformers import SentenceTransformer, util as transformer_util
from tqdm import tqdm

PARENT_DIR = Path(__file__).resolve().parent.parent
PMCID_PATH = PARENT_DIR / "pmc_ids.pkl"
EST_TIME = datetime.now(pytz.timezone("US/Eastern"))
IS_PEAK_HOURS = EST_TIME.weekday() < 5 and time(5, 0) <= EST_TIME.time() <= time(21, 0)

warnings.filterwarnings("ignore")

### TODO: Add ability for users to input their own set of PMC IDs to replace querying
### TODO: Add impact weights for articles using PubTator scores and/or number of citations
### TODO: Figure out why broad queries give "Error: 429"
### TODO: Use tags when applicable for Pubtator
### TODO: Save summaries to txt file
### TODO: Give option to go through fine tuning process (generate accuracy metrics, use all models) or not (add path to pretrained model)

def main():
    out_dir, query, max_articles, num_days, openai_key, anthropic_key, gemini_key = _parse_args(PARENT_DIR)

    summarizer = model.llm_summarizer(openai_key, anthropic_key, gemini_key)
    max_articles = _handle_num_requests(IS_PEAK_HOURS, max_articles)
    results_dir= Path(out_dir.joinpath(f"results_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"))
    results_dir.mkdir(parents=True, exist_ok=True)
    device = find_device()
    system_message = "You are an AI assistant tasked with summarizing articles. Your goal is to provide a concise, accurate, and informative summary of the key points in the given article text. Focus on capturing the main ideas, key findings, and important conclusions. Avoid including unnecessary details or tangents. The summary should be approximately 1-2 paragraphs in length."

    if not Path.exists(PMCID_PATH):
        process_pmcid_file(PMCID_PATH)
    with open(PMCID_PATH, "rb") as f:
        valid_pmc_ids = pickle.load(f)

    dict_articles = fetch_articles(query, max_articles, num_days, valid_pmc_ids)
    df_articles = _combine_article_sections(dict_articles)
    df_summaries, df_abstracts = summarize_articles(df_articles, device, summarizer, system_message)
    df_accuracies = quantify_accuracy(df_summaries, df_abstracts, results_dir)
    plot_accuracies(df_accuracies, results_dir)

    model_name_transformer, model_name_llm = find_best_models(df_accuracies)
    print(f"Best-performing transformer model: {model_name_transformer}")
    print(f"Best-performing LLM model:         {model_name_llm}")

    df_summaries_transformer = fine_tune(model_name_transformer, df_articles, device, results_dir)
    df_accuracies_transformer_finetune = quantify_accuracy(df_summaries_transformer, df_articles["abstract"], results_dir, finetune=True)
    plot_accuracies(df_accuracies_transformer_finetune, results_dir, finetune=True)


def _parse_args(parent_dir):
    """
    Store arguments passed by the user in the command line

    :return: query (str): \n
        Search term used to query articles
    :return: max_articles (int): \n
        Maximum number of articles to retrieve
    :return: num_days (int): \n
        Number of days prior to today to begin the search
    """

    parser = argparse.ArgumentParser(description="Research article summarizer")
    parser.add_argument('-o', '--out', type=str, default=parent_dir, help="Path to directory where outputs are saved.")
    parser.add_argument('-m', '--max_articles', type=int, default=100, help="Maximum number of articles to retrieve")
    parser.add_argument('-d', '--num_days', type=int, default=14, help="Number of days prior to today to begin the search")
    parser.add_argument('-ko', '--openai_key', type=str, default="", help="OpenAI API key")
    parser.add_argument('-ka', '--anthropic_key', type=str, default="", help="Anthropic API key")
    parser.add_argument('-kg', '--gemini_key', type=str, default="", help="Gemini API key")
    required = parser.add_argument_group('required arguments')
    required.add_argument('-q', '--query', help='Search term used to query articles', required=True)
    args = parser.parse_args()
    return args.out, args.query, args.max_articles, args.num_days, args.openai_key, args.anthropic_key, args.gemini_key


def _handle_num_requests(is_peak_hours, max_articles):
    """
    Ensure user does not exceed the 100-request limit during PMC peak hours

    Args:
        is_peak_hours (bool): True if the current time is during PMC peak hours, otherwise False
        max_articles (int): Maximum number of articles to retrieve

    :return: max_articles (int): \n
        Maximum number of articles to retrieve. Updated to 100 if during peak hours and was originally above 100
    """

    if is_peak_hours and max_articles > 100:
        print("Per https://pmc.ncbi.nlm.nih.gov/tools/oai/, PMC guidelines restrict scripts from making "
              "more than 100 requests during peak hours, defined as Monday-Friday from 5am - 9am EST.")
        print("Setting max_articles to 100...")
        return 100
    else:
        return max_articles


def process_pmcid_file(pmcid_path):
    """
    Fetch updated file with valid PMCIDs for Open Access articles and store them locally

    Args:
        pmcid_path (str): Path where PMCID file is saved locally
    """

    url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz"
    response = requests.get(url)

    with gzip.open(BytesIO(response.content), 'rt') as gz_file:
        df = pd.read_csv(gz_file, usecols=["PMCID"])

    set_ids = set(df["PMCID"].dropna())
    with open(pmcid_path, "wb") as f:
        pickle.dump(set_ids, f)


def fetch_articles(query, max_articles, num_days, valid_pmc_ids):
    """
    Store full-text articles over a specified period of time for a specified query.

    Args:
        query (str): Search term used to filter articles.
        max_articles (int): Maximum number of articles to retrieve.
        num_days (int): Number of days prior to today to begin the search.
        valid_pmc_ids (set): Open Access PMCIDs that are most likely to have full-text content.

    :return: articles (dict): \n
        Dictionary of full-text content for each PMCID.
    """

    print("Fetching articles now...")
    articles = {}
    page = 1
    date_end = datetime.now() - timedelta(days=0)
    date_start = date_end - timedelta(days=num_days)
    start_time = time_module.time()

    with tqdm(total=max_articles, desc="Fetching PMCIDs") as pbar:
        while len(articles) < max_articles:
            # Send a request to the API. Delay to comply with NCBI restrictions (3 rps)
            url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/?text={query}&page={page}&sort=date%20desc"
            response, start_time = utils._delayed_request(url, start_time)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()

                # Extract PMCIDs from the response
                pmcids = [result["pmcid"] for result in data["results"] if
                    "pmcid" in result
                    and result["pmcid"] not in articles.keys()
                    and result["pmcid"] in valid_pmc_ids
                    and (date_start <= datetime.strptime(result["date"], "%Y-%m-%dT%H:%M:%SZ") <= date_end)
                ]

                # Fetch full articles from PMCIDs
                if len(pmcids) > 0:
                    articles, start_time, pbar = utils.parse_articles(pmcids, articles, start_time, pbar)

                # If search has surpassed the specified date range, complete the search
                final_article_date = datetime.strptime(data["results"][-1]["date"], "%Y-%m-%dT%H:%M:%SZ")
                if final_article_date <= date_start:
                    pbar.update(max_articles - pbar.n)
                    return articles

                page += 1
            
            else:
                pbar.update(max_articles - pbar.n)
                print(f"Error: {response.status_code}")
                return articles
    
    print(f"{len(articles)} full-text articles found")
    return articles


def _combine_article_sections(dict_articles):
    """
    Combines the text from multiple columns into a single column for natural language processing

    Args:
        dict_articles (dict): Dictionary with keys as PMCIDs and values as dictionaries with section headers and corresponding text

    :return: df_articles *(pandas.DataFrame)*: \n
        Dataframe containing PMCIDs and corresponding full-text articles
    """

    print("Combining article sections into full-text articles now...")

    df_articles = pd.DataFrame(dict_articles).T.reset_index(names='pmcid')
    
    def combine_non_empty_sections(row):
        sections = [
            str(row["TITLE"]),
            str(row["INTRO"]),
            str(row["CASE"]),
            str(row["METHODS"]),
            str(row["RESULTS"]),
            str(row["DISCUSS"]),
            str(row["CONCL"]),
        ]
        non_empty_sections = [section for section in sections if section.strip() != ""]
        return "\n".join(non_empty_sections)
    
    df_articles["full_text"] = df_articles.apply(combine_non_empty_sections, axis=1)
    df_articles.rename({"ABSTRACT":"abstract"}, inplace=True, axis=1)
    print("Full-text article strings generated!")
    return df_articles[["pmcid", "abstract", "full_text"]]


def find_device():
    """
    Find GPU configuration for user's device

    :return: device *(int)*: \n
        0 if using GPUs, otherwise -1
    """

    if torch.backends.mps.is_available():
        device = 0
    elif torch.cuda.is_available():
        device = 0
    else:
        device = -1

    return device


def summarize_articles(df_articles, device, summarizer, system_message):
    """
    Summarize a full-text article using various NLP/LLM models

    Args:
        df_articles (pandas.DataFrame): Dataframe containing PMCIDs and corresponding full-text articles
        device (int): 0 if using GPUs, otherwise -1
        summarizer (research_summarizer.model.llm_summarizer): Summarizer object 
        system_message (str): Prompt to provide instructions for LLMs

    :return: summaries *(pandas.DataFrame)*: \n
        Dataframe containing PMCIDs and summaries from each model
    :return: abstracts *(pandas.DataFrame)*: \n
        Dataframe containing PMCIDs and corresponding abstracts
    """

    print("Generating summaries for each article now...")
    summaries = {}
    abstracts = {}
    for _, df_article in tqdm(df_articles.iterrows(), total=len(df_articles), desc="Summarizing articles"):
        summaries[df_article['pmcid']] = utils._summarize_article(
            df_article['full_text'], 
            device, 
            summarizer, 
            system_message,
        )
        abstracts[df_article['pmcid']] = [df_article['abstract']]
    df_summaries = pd.DataFrame(summaries)
    df_abstracts = pd.DataFrame(abstracts)
    print("Article summaries generated!")
    return df_summaries, df_abstracts


def quantify_accuracy(df_summaries, df_abstracts, results_dir, finetune=False):
    """
    Calculate cosine similarity between abstracts and corresponding summaries for each model

    Args:
        df_summaries (pandas.DataFrame): Mapping from PMCIDs to dictionary with summaries from each model
        df_abstracts (pandas.DataFrame): Mapping from PMCIDs to corresponding abstracts
        results_dir (pathlib.Path): Path to directory where outputs are saved
        finetune (bool): Plotting accuracies for fine-tuned model (Default: False)

    :return: df_accuracies *(pandas.DataFrame)*: \n
        Cosine similarity accuracies for summaries from each model for each article
    """

    print("Quantifying cosine similarity for each summary now...")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    accuracies = {}
    for pmc_id in df_summaries.keys():
        abstract = df_abstracts[pmc_id]
        abstract_encoding = sentence_model.encode(abstract)
        abstract_encoding = torch.tensor(abstract_encoding).clone().detach()
        accuracies[pmc_id] = {}
        for model_name in df_summaries[pmc_id].keys():
            summary = df_summaries[pmc_id][model_name]
            summary_encoding = sentence_model.encode(summary)
            summary_encoding = torch.tensor(summary_encoding).clone().detach()
            accuracy = transformer_util.cos_sim(
                summary_encoding / summary_encoding.norm(),
                abstract_encoding / abstract_encoding.norm(),
            ).item()
            accuracies[pmc_id][model_name] = accuracy
    df_accuracies = pd.DataFrame(accuracies)
    fname = f"accuracies{'_finetune' if finetune else ''}.csv"
    df_accuracies.to_csv(results_dir / fname)
    print(f"Article summaries saved to {str(results_dir / fname)}")
    return df_accuracies


def plot_accuracies(df_accuracies, results_dir, finetune=False):
    """
    Generate box plots to demonstrate accuracy for each model

    Args:
        df_accuracies (pandas.DataFrame): Cosine similarity accuracies for summaries from each model for each article
        results_dir (pathlib.Path): Path to directory where outputs are saved
        finetune (bool): Plotting accuracies for fine-tuned model (Default: False)
    """

    print("Plotting accuracy scores...")
    df_accuracies = df_accuracies.reset_index().rename(columns={"index": "Model"})
    df_accuracies = df_accuracies.melt(id_vars="Model", var_name="Article", value_name="Accuracy")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="Accuracy", data=df_accuracies, palette="Set3")
    plt.title("Model Accuracies Across Articles")
    plt.xlabel("Summarization Model")
    plt.ylabel("Accuracy")
    fname = f"accuracies{'_finetune' if finetune else ''}.png"
    plt.savefig(results_dir / fname, bbox_inches="tight")
    print(f"Plots saved to {str(results_dir / fname)}")


def find_best_models(df_accuracies):
    """
    Find the best-performing transformer and best-performing LLM

    Args:
        df_accuracies (pandas.DataFrame): Cosine similarity accuracies for summaries from each model for each article

    :return: model_name_transformer *(str)*: \n
        Name of the best-performing transformer model
    :return: model_name_llm *(str)*: \n
        Name of the best-performing LLM model
    """

    df_accuracies["Average"] = df_accuracies.mean(axis=1)
    df_accuracies = df_accuracies.sort_values(by="Average", ascending=False)

    try:
        df_accuracies_transformer = df_accuracies.loc[df_accuracies.index.isin(["Bart", "Falconsai", "BigBird"])]
        model_name_transformer = df_accuracies_transformer.index[0]
    except:
        model_name_transformer = ""

    try:
        df_accuracies_llm = df_accuracies.loc[df_accuracies.index.isin(["Llama", "Gemma", "Phi3"])]
        model_name_llm = df_accuracies_llm.index[0]
    except:
        model_name_llm = ""

    return model_name_transformer, model_name_llm


def fine_tune(model_name_transformer, df_articles, device, results_dir):
    """
    Fine-tune best-performing pretrained models using article abstracts

    Args:
        model_name_transformer (str): Name of the best-performing transformer model
        df_articles (pandas.DataFrame): Dataframe containing PMCIDs and corresponding full-text articles
        device (int): 0 if using GPUs, otherwise -1
        results_dir (pathlib.Path): Path to directory where outputs are saved

    :return: df_summaries_transformer *(pandas.DataFrame)*: \n
        Mapping from PMCIDs to summaries from the new fine-tuned model
    """

    df_summaries_transformer = model.fine_tune_transformer(model_name_transformer, df_articles, device, results_dir)
    return df_summaries_transformer
