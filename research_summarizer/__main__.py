import research_summarizer.utils as utils

def main():
    #out_dir = utils.parse_args()
    query, max_articles, email, num_days = utils.parse_args()
    
    # Fetch abstracts according to defined search query.
    pubmed_ids = utils.fetch_pubmed_ids(query, max_articles, email, num_days)
    abstracts = utils.fetch_pubmed_abstracts(pubmed_ids)

    # Print the first 10 abstracts, and the total number of abstracts
    print(f"{len(abstracts)} abstracts found")
    utils.print_abstracts(abstracts, 1)

    """
    # Example of using the fetched metadata to retrieve the full text
    full_text_links = utils.fetch_pubmed_links(pubmed_ids)
    for article in full_text_links:
        if article['FullTextLink']:
            print(f"Fetching full text for: {article['Title']}")
            article_body = utils.fetch_pubmed_fulltext(article['FullTextLink'])
            print(article_body)
        else:
            print(f"No free full-text link found for article: {article['Title']}")
    """
