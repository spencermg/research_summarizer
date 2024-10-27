import research_summarizer.utils as utils
import pandas as pd

def main():
    #out_dir = utils.parse_args()
    query, max_articles, num_days, pmc_id_path = utils.parse_args()

    # Load valid PMIDs to filter queries.
    valid_pmids = pd.read_csv(pmc_id_path)["PMID"].tolist()
    valid_pmcids = pd.read_csv(pmc_id_path)["PMCID"].tolist()
    pmids_dict = dict(zip(valid_pmids, valid_pmcids))

    # Fetch PubMed IDs according to defined search query.
    pmids = utils.fetch_ids_pubtator(query, max_articles, num_days, valid_pmids)
    print(f"{len(pmids)} PubMed IDs found")
    
    # Fetch full-text articles for each PubMed ID.
    articles, excluded_ids = utils.fetch_full_articles(pmids, pmids_dict)
    print(f"{len(articles)} full-text articles found")
    print(f"Excluded articles: {excluded_ids}")
    