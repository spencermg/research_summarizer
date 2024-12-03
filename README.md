# **LLM-Powered Biomedical Research Summarizer**

## Table of Contents 
#### [1. Background](#background)
#### [2. Installation](#installation)
#### [3. Examples](#examples)
#### [4. References](#references)


<a id="background"></a>
## **Background**
### **Project Overview**

This project aims to develop a platform that leverages the power of Large Language Models (LLMs) to summarize novel breakthroughs in biomedical research. By fine-tuning pre-trained LLMs, we aim to provide researchers and clinicians with concise and relevant summaries of the latest research papers, tailored to their specific interests.

**Key Features:**

* **Customizable Research Fields:** Users can define specific areas of interest, allowing for highly relevant summaries. This can range from broad fields like immunology to specific topics such as rare genomic variants linked with pancreatic cancer.  
* **Flexible Timeframes:** Specify the desired time range for research summaries (e.g., the last week, month, or year).  
* **Individual Paper Summaries:** Summarize single research papers of interest.  
* **Multi-Modal Output:** Summaries are provided in both text and audio format for accessibility.  
* **Source Linking:** Links to the original research papers are provided for easy access.  
* **LLM Comparison and User Preferences:** The platform will compare results from various pre-trained LLMs and allow users to indicate their preferences, ensuring personalized summaries.

### **Target Audience**

This tool is designed for:

* **Researchers:** Stay updated on the latest findings in their field without spending hours reading papers.  
* **Clinicians:** Quickly extract clinically relevant information from research papers.

### **Problem and Solution**

While many advanced LLMs exist, they often lack the dynamism and specialization needed to effectively summarize new research papers. This project addresses this gap by:

* **Dynamically updating** the knowledge base with the latest research publications.  
* **Specializing** the LLMs to the domain of biomedical research papers.

### **Data Sources**

We will utilize the following APIs to access research papers and their metadata:

* **arXiv API:** Access research papers and abstracts from arXiv.  
* **PubMed Central API:** Access full-text articles and associated metadata from PubMed Central.

This project will provide a valuable resource for healthcare professionals, enabling them to efficiently stay abreast of the latest advancements in their field.


<a id="installation"></a>
## **Installation**

Before installation, ensure you have Python and Anaconda installed on your computer. Once both are installed, open a command line interface and clone this repository directly from github:

`git clone https://github.com/spencermg/research_summarizer.git`

Once cloned, navigate to the research_summarizer directory...

- On Mac/Linux:

`cd path_to_research_summarizer_directory`

- On Windows:

`chdir path_to_research_summarizer_directory`

After navigating to the research_summarizer directory, create an anaconda environment using Python 3.11 and install dependencies:

```shell
# Create the anaconda environment
conda create -n research_summarizer python==3.11

# Activate the environment you just created
conda activate research_summarizer

# Install dependencies
pip install ./
```

<a id="examples"></a>
## **Examples**

You are now ready to generate summaries of recent advancements in the scientific literature! Here are a few examples of how can run this program:

```shell
# Summarize up to 5 Alzheimer's genomics articles over the last 7 days using pretrained transformer models
summarize -q "Alzheimers genomics" -d 7 -m 5

# Summarize up to 10 lung histology articles over the last 3 days using pretrained transformer models and save outputs to the summarization_results directory on your Desktop
summarize -q "Lung histology" -d 3 -m 10 -o Desktop/summarization_results

# Summarize up to 20 cancer immunotherapy articles over the last 30 days using both pretrained transformer models and LLM models
summarize -q "Cancer immunotherapy" -d 30 -m 20 --openai_key your_openai_key --anthropic_key your_anthropic_key --gemini_key your_gemini_key
```

<a id="referenecs"></a>
## **References**

1.	Hanson, M., Barriero, P., Crosetto, P., Brockington, D. (2023). The strain on scientific publishing. arXiv preprint arXiv 2309.15884. https://doi.org/10.48550/arXiv.2309.15884
2.	Mishra, R., Bian, J., Fiszman, M., Weir, C. R., Jonnalagadda, S., Mostafa, J., & Del Fiol, G. (2014). Text summarization in the biomedical domain: a systematic review of recent research. Journal of biomedical informatics, 52, 457–467. https://doi.org/10.1016/j.jbi.2014.06.009
3.	Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., & Goharian, N. (2018). A discourse-aware attention model for abstractive summarization of long documents. arXiv preprint arXiv:1804.05685. https://arxiv.org/pdf/1804.05685
4.	U.S. National Library of Medicine. (2024, September 10). PMC Policies. National Center for Biotechnology Information. https://www.ncbi.nlm.nih.gov/pmc/about/guidelines/
5.	About arxiv. About arXiv - arXiv info. (n.d.). https://info.arxiv.org/about/index.html
6.	Kirmani, M., Kour, G., Mohd, M. et al. (2024). Biomedical semantic text summarizer. BMC Bioinformatics 25, 152. https://doi.org/10.1186/s12859-024-05712-x
7.	Bajaj, A., Dangati, P., Krishna, K., et al. (2021). Long Document Summarization in a Low Resource Setting using Pretrained Language Models. arXiv preprint arXiv 2103.00751. https://doi.org/10.48550/arXiv.2103.00751
8.	Sinha, A., Yadav, A., Gahlot, A. (2018). Extractive Text Summarization using Neural Networks. arXiv preprint arXiv 1802.18137. https://doi.org/10.48550/arXiv.1802.18137
9.	Liu, Z., Chen, N. (2021). Dynamic Sliding Window for Meeting Summarization. arXiv preprint arXiv 2108.13629. https://doi.org/10.48550/arXiv.2108.13629
10.	Zdeb, K. K. (2024b, April 28). Towards infinite LLM context windows. Medium. https://towardsdatascience.com/towards-infinite-llm-context-windows-e099225abaaf
11.	Zhang, H., Yu, P.S., and Zhang, J. (2024). A Systematic Survey of Text Summarization: From Statistical Methods to Large Language Models. arXiv preprint arXiv:1804.05685. https://doi.org/10.48550/arXiv.2406.11289
12.	Wei, C., Allot, A., Lai, P., Leaman, R., Tian, S., Luo, L., Jin, Q., Wang, Z., Chen, Q., Lu, Z. (2024). PubTator 3.0: an AI-powered literature resource for unlocking biomedical knowledge. Nucleic Acids Research, Volume 52, Issue W1, Pages W540–W546, https://doi.org/10.1093/nar/gkae235
13.	https://huggingface.co/facebook/bart-large-cnn
14.	https://huggingface.co/Falconsai/medical_summarization
15.	https://huggingface.co/google/bigbird-pegasus-large-pubmed
16.	Haque, S., Eberhart, Z., Bansal, A., McMillan, C. (2022). Semantic similarity metrics for evaluating source code summarization. Proceedings of the 30th IEEE/ACM International Conference on Program Comprehension, pp. 36–47.
17.	https://huggingface.co/gradio
18.	Wu, S., et al. (2024). "Retrieval-augmented generation for natural language processing: A survey." arXiv preprint arXiv:2407.13193. https://doi.org/10.48550/arXiv.2407.13193
