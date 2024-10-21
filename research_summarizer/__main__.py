import research_summarizer.utils as utils

def main():
    #out_dir = utils.parse_args()
    query, max_articles, email, num_days = utils.parse_args()

    # Fetch PubMed IDs and abstracts according to defined search query.
    pmids = utils.fetch_ids_pubtator(query, max_articles, num_days)

    print(f"{len(pmids)} articles found")
    print(f"{len(set(pmids))} unique articles found")

    #abstracts, ids_removed = utils.fetch_abstracts_pubtator(pmids)
    #pmids = [int(pmid) for pmid in pmids if pmid not in ids_removed]
    
    # Print the first 2 abstracts, and the total number of abstracts
    #print(f"{len(pmids)} abstracts fetched")
    #utils.print_abstracts(abstracts, 2)


    """
    # Example of using the fetched metadata to retrieve the full text
    full_text_links = utils.fetch_pubmed_links(pmids)
    for article in full_text_links:
        if article['FullTextLink']:
            print(f"Fetching full text for: {article['Title']}")
            article_body = utils.fetch_pubmed_fulltext(article['FullTextLink'])
            print(article_body)
        else:
            print(f"No free full-text link found for article: {article['Title']}")
    """
