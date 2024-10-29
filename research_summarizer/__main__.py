import research_summarizer.utils as utils
from pathlib import Path
import pickle
from datetime import datetime, time
import pytz

PMCID_PATH = Path(__file__).resolve().parent.parent / "pmc_ids.pkl"
EST_TIME = datetime.now(pytz.timezone("US/Eastern"))
IS_PEAK_HOURS = EST_TIME.weekday() < 5 and time(5, 0) <= EST_TIME.time() <= time(21, 0)

### TODO: Add ability for users to input their own set of PMC IDs to replace querying
### TODO: Add impact weights for articles using PubTator scores and/or number of citations
### TODO: Figure out why broad queries give "Error: 429"

def main():
    query, max_articles, num_days = utils._parse_args()
    max_articles = utils._handle_num_requests(IS_PEAK_HOURS, max_articles)

    if not Path.exists(PMCID_PATH):
        utils.process_pmcid_file(PMCID_PATH)
    
    with open(PMCID_PATH, "rb") as f:
        valid_pmc_ids = pickle.load(f)

    # Fetch articles according to defined search query.
    articles = utils.fetch_articles(query, max_articles, num_days, valid_pmc_ids)
    print(f"{len(articles)} full-text articles found")
    