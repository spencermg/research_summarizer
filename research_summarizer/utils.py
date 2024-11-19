import argparse
import requests
from datetime import datetime, timedelta
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm
import gzip
from io import BytesIO
import pandas as pd
import pickle
import torch


SECTIONS_OF_INTEREST = ["TITLE", "ABSTRACT", "INTRO", "CASE", "METHODS", "RESULTS", "DISCUSS", "CONCL"]


def _parse_args(parent_dir):
    """
    Store arguments passed by the user in the command line.

    :return: query (str): \n
        Search term used to query articles.
    :return: max_articles (int): \n
        Maximum number of articles to retrieve.
    :return: num_days (int): \n
        Number of days prior to today to begin the search.
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
    Ensure user does not exceed the 100-request limit during PMC peak hours.

    Args:
        is_peak_hours (bool): True if the current time is during PMC peak hours, otherwise False.
        max_articles (int): Maximum number of articles to retrieve.

    :return: max_articles (int): \n
        Maximum number of articles to retrieve. Updated to 100 if during peak hours and was originally above 100.
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
    Fetch updated file with valid PMCIDs for Open Access articles and store them locally.

    Args:
        pmcid_path (str): Path where PMCID file is saved locally.
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

    articles = {}
    page = 1
    date_end = datetime.now() - timedelta(days=0)
    date_start = date_end - timedelta(days=num_days)
    start_time = time.time()

    with tqdm(total=max_articles, desc="Fetching PMCIDs") as pbar:
        while len(articles) < max_articles:
            # Send a request to the API. Delay to comply with NCBI restrictions (3 rps)
            url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/?text={query}&page={page}&sort=date%20desc"
            response, start_time = _delayed_request(url, start_time)

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
                    articles, start_time, pbar = parse_articles(pmcids, articles, start_time, pbar)

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
    
    return articles


def parse_articles(pmcids, articles, start_time, pbar):
    """
    Retrieve full-text articles corresponding to each PubMed ID of interest.

    Args:
        pmcids (list): List of PubMed Central IDs.
        articles (dict): Current dictionary of full-text content for each PMCID.
        start_time (float): Time of most recent NCBI query.
        pbar (tqdm.std.tqdm): Progress bar tracking articles that have been retrieved.

    :return: articles *(dict)*: \n
        Dictionary of PMCIDs and corresponding full-text data with current article included. Full-text data are \n
        stored as dictionaries of section names and corresponding lists containing each paragraph in that section.
    """
    
    # Loop through each article.
    for i in range(len(pmcids)):
        url_pmc = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmcids[i]}/unicode"
        article, start_time, pbar = parse_article(url_pmc, start_time, pbar)
        if article is not None:
            articles[pmcids[i]] = article

        # Finish once the max number of articles has been reached.
        if len(articles) == pbar.total:
            break
        
    return articles, start_time, pbar


def parse_article(url_pmc, start_time, pbar):
    """
    Store full-text data from a specified URL.

    Args:
        url_pmc (str): URL for an article.
        start_time (float): Time of most recent NCBI query.
        pbar (tqdm.std.tqdm): Progress bar tracking articles that have been retrieved.

    :return: article *(dict)*: \n
        Contents of a single article, with section types as keys and strings of section contents as values.
    :return: start_time *(float)*: \n
        Time of most recent NCBI query, updated to account for the current article.
    :return: pbar *(tqdm.std.tqdm)*: \n
        Progress bar tracking articles that have been retrieved, updated to account for the current article.
    """

    response, start_time = _delayed_request(url_pmc, start_time)
        
    try:
        # Parse the XML content
        xml_data = response.content
        root = ET.fromstring(xml_data)
        
        # Extract full-text passages
        document = root.find(".//document")
        section_names = [section_name.text for section_name in document.findall(".//infon[@key='section_type']")]
        if "ABSTRACT" in section_names and sum([section in section_names for section in SECTIONS_OF_INTEREST]) >= 3:
            article = {}
            for section_name in SECTIONS_OF_INTEREST:
                article[section_name] = []

            for passage in document.findall(".//passage"):
                section_name = passage.find(".//infon[@key='section_type']").text
                if section_name in SECTIONS_OF_INTEREST:
                    article[section_name].append(passage.find("text").text.strip())
            pbar.update(1)
        else:
            return None, start_time, pbar

    except:
        return None, start_time, pbar
    
    for key in article.keys():
        article[key] = "\n".join(article[key])

    return article, start_time, pbar


def _delayed_request(url, start_time):
    """
    Query NCBI database ensuring compliance with 3rps limit.

    Args:
        url (str): URL fetching data from.
        start_time (float): Time of most recent NCBI query.

    :return: response (str): \n
        Data from specified URL.
    :return: start_time *(float)*: \n
        Updated start time making the start of the current query.
    """

    # Fetch current article. Delay to comply with NCBI restrictions (3 rps)
    time.sleep(max(1./3. - (time.time() - start_time), 0))
    start_time = time.time()
    response = requests.get(url)
    return response, start_time


def _combine_article_sections(dict_articles):
    """
    Combines the text from multiple columns into a single column for natural language processing.

    Args:
        dict_articles (dict): Dictionary with keys as PMCIDs and values as dictionaries with section headers and corresponding text

    :return: df_articles *(pandas.DataFrame)*: \n
        Dataframe containing PMCIDs and corresponding full-text articles.
    """

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
        df_articles (pandas.DataFrame): Dataframe containing PMCIDs and corresponding full-text articles.
        device (int): 0 if using GPUs, otherwise -1
        summarizer (research_summarizer.model.llm_summarizer): Summarizer object 
        system_message (str): Prompt to provide instructions for LLMs

    :return: summaries *(dict)*: \n
        Mapping from PMCIDs to dictionary with summaries from each model
    :return: abstracts *(dict)*: \n
        Mapping from PMCIDs to corresponding abstracts
    """

    summaries = {}
    abstracts = {}
    for _, df_article in df_articles.iterrows():
        summaries[df_article['pmcid']] = _summarize_article(
            df_article['full_text'], 
            device, 
            summarizer, 
            system_message,
        )
        abstracts[df_article['pmcid']] = [df_article['abstract']]
    df_summaries = pd.DataFrame(summaries)
    df_abstracts = pd.DataFrame(abstracts)
    return df_summaries, df_abstracts


def _summarize_article(article_text, device, summarizer, system_message):
    """
    Summarize a full-text article using various NLP/LLM models

    Args:
        article_text (str): Full text from an article
        device (int): 0 if using GPUs, otherwise -1
        summarizer (research_summarizer.model.llm_summarizer): Summarizer object 
        system_message (str): Prompt to provide instructions for LLMs

    :return: summaries *(str)*: \n
        Summarization of the full-text article
    """

    summaries = {}
    summaries = summarizer.summarize_bart(article_text, summaries, device)
    summaries = summarizer.summarize_falcons(article_text, summaries, device)
    summaries = summarizer.summarize_bigbird(article_text, summaries, device)
    summaries = summarizer.summarize_gpt(article_text, system_message, summaries)
    summaries = summarizer.summarize_anthropic(article_text, system_message, summaries)
    summaries = summarizer.summarize_gemini(article_text, system_message, summaries)
    return summaries
