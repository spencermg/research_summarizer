import requests
import time
import xml.etree.ElementTree as ET


SECTIONS_OF_INTEREST = ["TITLE", "ABSTRACT", "INTRO", "CASE", "METHODS", "RESULTS", "DISCUSS", "CONCL"]


def parse_articles(pmcids, articles, start_time, pbar):
    """
    Retrieve full-text articles corresponding to each PubMed ID of interest

    Args:
        pmcids (list): List of PubMed Central IDs
        articles (dict): Current dictionary of full-text content for each PMCID
        start_time (float): Time of most recent NCBI query
        pbar (tqdm.std.tqdm): Progress bar tracking articles that have been retrieved

    :return: articles *(dict)*: \n
        Dictionary of PMCIDs and corresponding full-text data with current article included. Full-text data are \n
        stored as dictionaries of section names and corresponding lists containing each paragraph in that section
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
    Store full-text data from a specified URL

    Args:
        url_pmc (str): URL for an article
        start_time (float): Time of most recent NCBI query
        pbar (tqdm.std.tqdm): Progress bar tracking articles that have been retrieved

    :return: article *(dict)*: \n
        Contents of a single article, with section types as keys and strings of section contents as values
    :return: start_time *(float)*: \n
        Time of most recent NCBI query, updated to account for the current article
    :return: pbar *(tqdm.std.tqdm)*: \n
        Progress bar tracking articles that have been retrieved, updated to account for the current article
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
    Query NCBI database ensuring compliance with 3rps limit

    Args:
        url (str): URL fetching data from
        start_time (float): Time of most recent NCBI query

    :return: response (str): \n
        Data from specified URL
    :return: start_time *(float)*: \n
        Updated start time making the start of the current query
    """

    # Fetch current article. Delay to comply with NCBI restrictions (3 rps)
    time.sleep(max(1./3. - (time.time() - start_time), 0))
    start_time = time.time()
    response = requests.get(url)
    return response, start_time


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
    summaries = summarizer.summarize_llm(article_text, system_message, summaries, "Llama")
    summaries = summarizer.summarize_llm(article_text, system_message, summaries, "Gemma")
    summaries = summarizer.summarize_llm(article_text, system_message, summaries, "Phi3")
    return summaries
