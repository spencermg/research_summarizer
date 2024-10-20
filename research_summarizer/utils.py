import argparse
from Bio import Entrez
import requests
from datetime import date, timedelta
import time
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description="Research article summarizer")
    #parser.add_argument('--out', type=str, default="output", help="Path to directory where outputs are saved.")
    parser.add_argument('-m', '--max_articles', type=int, default=1000, help="Maximum number of abstracts to analyze")
    parser.add_argument('-d', '--num_days', type=int, default=7, help="Number of days prior to today to begin the search")
    required = parser.add_argument_group('required arguments')
    required.add_argument('-e', '--email', help='User email address used for querying', required=True)
    required.add_argument('-q', '--query', help='Search term to query research topics', required=True)
    args = parser.parse_args()
    return args.query, args.max_articles, args.email, args.num_days

def print_abstracts(abstracts_list, num_abstracts):
    """
    Print out a sampling of abstracts from a list of abstracts.

    :param abstracts_list: Full list of abstracts.
    :param num_abstracts: Number of abstracts being printed from the full list.
    """
    for i, abstract in enumerate(abstracts_list[:num_abstracts]):
        print(f"Abstract {i+1}: {abstract}")

def fetch_ids_pubmed(query, max_results, email, num_days):
    """
    Retrieve PubMed IDs that match the given query.

    :param query: Word or phrase by which to query articles.
    :param max_results: Maximum number of articles to fetch.
    :param email: Email address used for article querying.
    :param num_days: Number of days prior to today to begin the search.

    :return: List of all PubMed IDs that were fetched.
    """

    Entrez.email=email
    date_end = date.today() - timedelta(days=7)
    date_start = date_end - timedelta(days=num_days)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, mindate=date_start, maxdate=date_end)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    return id_list

### TODO: Add functionality to filter by date
def fetch_ids_pubtator(query, max_results, num_days):
    # Define the base URL for the PubTator3 search API
    pmids = []
    page = 1

    while len(pmids) < max_results:
        # Send a GET request to the API
        url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/search/?text={query}&page={page}&sort=_id%20desc"
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract PMIDs from the response
            pmids += [str(result['pmid']) for result in data['results']]

            # Delay to comply with PubTator limits before going to the next page
            time.sleep(0.334)
            page += 1
        
        else:
            print(f"Error: {response.status_code}")
            return pmids
    
    return pmids

def fetch_abstracts_pubmed(pubmed_ids, batchsize=1000):
    """
    Retrieve abstracts corresponding to each PubMed ID of interest.

    :param pubmed_ids: List of PubMed IDs.
    :param batchsize: Maximum number of IDs per batch (Default: 1000).

    :return: List of abstracts fetched from the PubMed IDs of interest.
    """

    abstracts_list = []

    # Loop through each batch.
    for i in range(0, len(pubmed_ids), batchsize):
        j = i + batchsize

        # Check if the number of remaining IDs is less than the batch size.
        if j >= len(pubmed_ids):
            j = len(pubmed_ids)

        # Isolate abstracts from articles being fetched.
        handle = Entrez.efetch(db="pubmed", id=','.join(pubmed_ids[i:j]), rettype="xml", retmode="text", retmax=batchsize)
        records = Entrez.read(handle)
        abstracts = [pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
                     for pubmed_article in records['PubmedArticle']
                     if 'Abstract' in pubmed_article['MedlineCitation']['Article'].keys()]
        abstracts_list += abstracts
    return abstracts_list

def fetch_abstracts_pubtator(pmids, batchsize=10):
    """
    Retrieve abstracts corresponding to each PubMed ID of interest.

    :param pmids: List of PubMed IDs.
    :param batchsize: Maximum number of IDs per batch (Default: 1000).

    :return: List of abstracts fetched from the PubMed IDs of interest.
    """
        
    # List to store the abstracts
    abstracts = []

    # Keep track of pmids that do not have abstracts
    ids_removed = []
    
    # Loop through each batch.
    for i in range(0, len(pmids), batchsize):
        # Check if the number of remaining IDs is less than the batch size.
        j = i + batchsize
        if j >= len(pmids):
            j = len(pmids)

        # Join the PMIDs with commas to form the URL parameter
        pmid_str = ",".join(pmids[i:j])
        url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml?pmids={pmid_str}"
        
        # Send a GET request to fetch the XML data
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the XML content
            xml_data = response.content
            root = ET.fromstring(xml_data)
            
            # Find and extract all abstract passages
            for document in root.findall(".//document"):
                for passage in document.findall(".//passage"):
                    infon_type = passage.find(".//infon[@key='type']").text
                    if infon_type == "abstract":
                        try:
                            abstract_text = passage.find("text").text
                            abstracts.append(abstract_text)
                        except Exception as e:
                            pmid = document.find(".//id").text
                            pmid_url = f"https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocxml?pmids={pmid}"
                            ids_removed.append(pmid)
                            print(f"Skipping \"{pmid_url}\" with exception: {e}")

            # Delay to comply with PubTator limits.
            time.sleep(0.334)
        else:
            print(f"Failed to fetch XML. Status code: {response.status_code}")
        
    return abstracts, ids_removed

'''
def fetch_pubmed_links(pubmed_ids, batch_size=1000):
    """
    Fetch PubMed metadata to check for free full-text availability.

    :param pubmed_ids: List of PubMed IDs.
    :param batch_size: Maximum number of IDs per batch (Default: 1000).

    :return: List of dictionaries with PubMed ID, title, and full-text links (if available).
    """
    full_text_links = []

    # Loop through PubMed IDs in batches
    for i in range(0, len(pubmed_ids), batch_size):
        j = i + batch_size
        if j >= len(pubmed_ids):
            j = len(pubmed_ids)

        handle = Entrez.efetch(db="pubmed", id=','.join(pubmed_ids[i:j]), rettype="xml", retmode="text")
        records = Entrez.read(handle)

        for pubmed_article in records['PubmedArticle']:
            # Check for full-text links in PubmedData
            pmid = pubmed_article['MedlineCitation']['PMID']
            title = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
            free_text_link = None
            if 'PubmedData' in pubmed_article and 'ArticleIdList' in pubmed_article['PubmedData']:
                for article_id in pubmed_article['PubmedData']['ArticleIdList']:
                    if article_id.attributes.get('IdType') == 'doi':
                        free_text_link = f"https://doi.org/{article_id}"
                    elif article_id.attributes.get('IdType') == 'pmc':
                        free_text_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{article_id}/"

            # Append the full-text link if available
            ### TODO: Move this inside conditionals above?
            full_text_links.append({
                'PMID': pmid,
                'Title': title,
                'FullTextLink': free_text_link
            })

    return full_text_links

def fetch_pubmed_fulltext(pmc_url):
    """
    Fetch the full text of an article from PubMed Central (PMC).

    :param pmc_url: URL of the article on PubMed Central.
    :return: Full-text content of the article.
    """
    response = requests.get(pmc_url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract full-text content (usually under <body> or specific sections like <sec>)
        article_body = soup.find_all('body')  # Adapt this based on the article's HTML structure

        if article_body:
            return article_body[0].get_text()
        else:
            return "Full text not found."
    else:
        return f"Failed to fetch article. Status code: {response.status_code}"
'''
