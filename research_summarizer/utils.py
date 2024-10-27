import argparse
import requests
from datetime import datetime, timedelta
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm

SECTIONS_OF_INTEREST = ["TITLE", "ABSTRACT", "INTRO", "CASE", "METHODS", "RESULTS", "DISCUSS", "CONCL"]

def parse_args():
    parser = argparse.ArgumentParser(description="Research article summarizer")
    #parser.add_argument('--out', type=str, default="output", help="Path to directory where outputs are saved.")
    parser.add_argument('-m', '--max_articles', type=int, default=100, help="Maximum number of abstracts to analyze")
    parser.add_argument('-d', '--num_days', type=int, default=14, help="Number of days prior to today to begin the search")
    required = parser.add_argument_group('required arguments')
    required.add_argument('-q', '--query', help='Search term to query research topics', required=True)
    required.add_argument('-p', '--pmc_id_path', help='Path to .csv file containing two columns: \"PMID\", which contains PMIDs, and \"PMCID\", which contains PMCIDs.', required=True)
    args = parser.parse_args()
    return args.query, args.max_articles, args.num_days, args.pmc_id_path


def fetch_ids_pubtator(query, max_results, num_days, valid_pmids):
    pmids = []
    page = 1
    date_end = datetime.now() - timedelta(days=0)
    date_start = date_end - timedelta(days=num_days)
    start_time = None

    with tqdm(total=max_results, desc="Fetching PMIDs") as pbar:
        while len(pmids) < max_results:
            # Send a GET request to the API. Delay to comply with NCBI restrictions (3 rps)
            url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/?text={query}&page={page}&sort=date%20desc"
            if start_time is not None:
                time.sleep(max(1./3. - (time.time() - start_time), 0))
            start_time = time.time()
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                data = response.json()

                # Extract PMIDs from the response
                pmids += [
                    result['pmid'] for result in data['results'] 
                        if (date_start <= datetime.strptime(result["date"], "%Y-%m-%dT%H:%M:%SZ") <= date_end) 
                        and result['pmid'] not in pmids
                        and result['pmid'] in valid_pmids
                ]

                pbar.update(len(pmids) - pbar.n)
                page += 1
            
            else:
                print(f"Error: {response.status_code}")
                return pmids
    
    return pmids[:max_results]


def fetch_full_articles(pmids, pmids_dict):
    """
    Retrieve full-text articles corresponding to each PubMed ID of interest.

    :param pmids: List of PubMed IDs.

    :return articles: Dictionary of PubMed IDs and corresponding full-text data. Full-text data are stored
    as dictionaries of section names and corresponding lists containing each paragraph in that section.
    """
        
    # Store articles
    articles = {}
    excluded_ids = []
    
    # Loop through each article.
    for i in tqdm(range(len(pmids)), desc="Fetching Articles"):
        # Fetch current article. Delay to comply with NCBI restrictions (3 rps)
        url_pmc = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmids_dict[pmids[i]]}/unicode"
        if i > 0:
            time.sleep(max(1./3. - (time.time() - start_time), 0))
        start_time = time.time()
        response = requests.get(url_pmc)
        
        # Check if the request was successful
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
                        article[section_name].append(passage.find("text").text)

                articles[pmids[i]] = article

            else:
                excluded_ids.append(pmids_dict[pmids[i]])

        except Exception as e:
            excluded_ids.append(pmids_dict[pmids[i]])

    return articles, excluded_ids
