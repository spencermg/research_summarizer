# **LLM-Powered Biomedical Research Summarizer**

## Table of Contents 
#### [1. Background](#background)
#### [2. References](#references)


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


<a id="referenecs"></a>
## References

1. https://www.ncbi.nlm.nih.gov/pmc/about/guidelines/ 
2. https://info.arxiv.org/about/index.html
3. Mishra, R., Bian, J., Fiszman, M., Weir, C. R., Jonnalagadda, S., Mostafa, J., & Del Fiol, G. (2014). Text summarization in the biomedical domain: a systematic review of recent research. Journal of biomedical informatics, 52, 457–467. https://doi.org/10.1016/j.jbi.2014.06.009
4. Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., & Goharian, N. (2018). A discourse-aware attention model for abstractive summarization of long documents. arXiv preprint arXiv:1804.05685. https://arxiv.org/pdf/1804.05685
5. Kirmani, M., Kour, G., Mohd, M. et al. Biomedical semantic text summarizer. BMC Bioinformatics 25, 152 (2024). https://doi.org/10.1186/s12859-024-05712-x
6. Chih-Hsuan Wei, Alexis Allot, Po-Ting Lai, Robert Leaman, Shubo Tian, Ling Luo, Qiao Jin, Zhizheng Wang, Qingyu Chen, Zhiyong Lu, PubTator 3.0: an AI-powered literature resource for unlocking biomedical knowledge, Nucleic Acids Research, Volume 52, Issue W1, 5 July 2024, Pages W540–W546, https://doi.org/10.1093/nar/gkae235
7. Comeau DC, Wei CH, Islamaj Doğan R, and Lu Z. PMC text mining subset in BioC: about 3 million full text articles and growing, Bioinformatics, btz070, 2019, https://doi.org/10.1093/bioinformatics/btz070
