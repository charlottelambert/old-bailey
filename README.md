## Senior Thesis Project

### Code contained in this project

The following descriptions relate to code in the `thesis` directory. To get more information, you see `README.md` files in each sub-directory. Run each file with the `--help` flag to get more info for each program.

Name | Content
-------|-------
`analyze` | Code for analyzing processed data
`data` | Code for acquiring and processing text data
`lambert-final-thesis.pdf` | Final thesis draft.
`logs` | Directory containing logfiles from topic modeling
`README.md` | This file
`topic-modeling` | Code for running LDA/DTM/other variations of topic modeling
`utils.py` | Functions used by code in several directories
`vector-space` | Code for running vector space models

To obtain the data used in this project, you will need two different datasets. For consistency with the instructions included in the `README.md` files, directory names are specified for each dataset:
1. [The Proceedings of Old Bailey](https://www.oldbaileyonline.org/index.jsp): Download the data from [here](https://figshare.shef.ac.uk/articles/Old_Bailey_Online_XML_Data/4775434) and store the XML files in a directory called `sessionsPapers`.
2. The Ordinarys Accounts: The link in for downloading the Old Bailey data will also download the Ordinarys Accounts. Make sure and store the XML files in a directory with the name `ordinarysAccounts`.
3. London Lives (optional): The [London Lives](https://www.londonlives.org) dataset is supplementary, but you can download it [here](https://figshare.com/articles/London_Lives_XML_Data/4797829) and save it into a directory called `londonLives`. The structure is different than that of the previous two datasets, simply include all the subdirectories as is.

Once the data is obtained, you can combine the first two datasets into a directory called `sessionsAndOrdinarys` so all the Old Bailey data is together and can be processed at one time. Use this command:

```
mkdir sessionsAndOrdinarys
cp sessionsPapers/* sessionsAndOrdinarys
cp ordinarysAccounts/* sessionsAndOrdinarys
```

### Information about this project

The Proceedings of the Old Bailey, 1674–1913 (Hitchcock et al., 2012b) is a published record of criminal proceedings at London’s central criminal court. The Proceedings primarily depict the lives of the "non-elite" population of London. This project explores these proceedings to study this specific population over the approximately 250-year time period of the publication. Because the corpus spans a significant period of history, it can be examined to identify evolving patterns related to different social groups represented in the text. This project aims to identify which computational methods can reveal interesting sociolinguistic information about this corpus. More specifically, this paper will explore unsupervised techniques like latent Dirichlet allocation (LDA) (Blei et al., 2003), Word2Vec (Mikolov et al., 2013), and Embedded Topic Modeling (ETM) (Dieng et al., 2019b) when applied to the Proceedings of Old Bailey. Additionally, temporal variants of these methods, such as Dynamic Topic Modeling (DTM) (Blei and Lafferty, 2006), Dynamic Embedded Topic Modeling (DETM) (Dieng et al., 2019a), and LDA and Word2Vec manually run across different time slices, are applied to analyze the corpus over time.

