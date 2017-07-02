# biblio-duplicate-detection
This repo deals with the problem of detecting near-duplicates among Russian-language bibliographic references. For the purpose of obtaining references, an additional task is solved — the allocation of bibliographic references from scientific documents. To build the base of unique links, a search engine indexing is implemented.

## Structure of project
The `notebooks` directory has jupyter-notebook analogues of py-files in `py` directory.

The experiment provides two steps. 

First step is learning on the data labeled by assessors. Texts of scientific papers were divided to different directories by means of which popular scientific work they cite. So papers in one folder should definitely have the reference on that popular work. Assessors took away everything but bibliography, deleted hyphenation and marked reference that should be in common between all papers in folder. A ML algorithm learns which pairs are marked as duplicates. 

Second step - extracting references from raw text file automatically and labeling them (1/0 - duplicates or not). Performance of automatic labeling is counted by sampling and measuring the share of right labels.

`create_train_test_labeled` - collecting the data from text files with references, extracting features

`create_train_test_unlabeled` - collecting the data from text files with full raw text, extracting references and features

`random_forest_labeled_data` - learning RF, measuring the performance

`experiments` - building the index structure to store unique references, experimenting with size of n-gram

`preprocess_observation` - boxed preprocessing steps for pair of references to compare for using in index experiments.
