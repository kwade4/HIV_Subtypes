## HIV-1 Subtype Classification 

#### Background 

* HIV-1 is classified into 4 groups: M, N, O, and P, and of these groups, group M is the most widespread and clinically relevant. 
* Group M is subdivided into 9 distinct subtypes: A, B, C, D, F, G, H, J, K. 
* 2 or more subtypes can combine to form a hybrid form called a circulating recombinant form (CRFs). 
* Rates of disease progression vary among HIV-1 subtypes, with some subtypes showing increased drug resistance and virulence. 
* Understanding the distribution of HIV-1 subtypes is crucial for the development of vaccines and the clinical management of HIV-1 infections.  For infected individuals, classifying HIV-1 subtype is a crucial step in clinical infection management.  
* HIV-1 subtyping = classifying HIV-1 into subtypes. 

#### Research Gap 
* Most HIV-1 subtype classification methods rely on aligning an input sequence against a set of pre-defined subtype reference sequences. These alignment-based methods are computationally expensive, especially for long sequences and rely on ad hoc parameter settings, which can limit reproducibility. 
* To address the limitations of alignment-based methods, alignment-free methods have been developed. 
* Few studies (if any) have compared DNA vectorization methods for HIV-1 subtyping. 
* Few studies (if any) have applied multi-task learning to HIV-1 subtype classification. 

#### Possible Research Objectives 
* To characterize the effect of DNA sequence vectorization methods on HIV-1 subtyping. 
* Apply multi-task learning to HIV-1 subtype classification. 
* Develop an improved HIV-1 subtype classification method. 

#### Methodology 

1. Explore different ways to vectorize DNA sequences: 
    * **Word2Vec** 
    * **K-mers** (of varying length)
    * **Subsequence natural vector**: includes number and distrubution of nucleotides, position, and variance 
    * **Natural vector**: number and distribtion of nucleotides
    * Other language model encodings
 
2. Feature selection:
    * LASSO
    * PCA
    * Pre-trained CNN (as Kyle suggested)  

3. Machine Learning & Deep Learning  
    * Classical Machine Learning 
        * SVM (one-vs-rest)
        * Multi-class logistic regression 
        * XGBoost
        * LASSO
        * Naive Bayes?  
        * KNN clustering 
    * Deep Learning 
        * 1D-CNN 
    * Multi-task learning (time-permitting)
    
4. Evaluation Metrics 
    * Confusion Matrix
    * Accuracy, precision (macro-precision), recall, F1-score (macro F1-score)
    * AUROC, AUPRC
    * Cohen's Kappa 
    
At our meeting, we had discussed evaluating our models with and without applying feature selection. 


#### Distribution of Tasks 

##### Experiments 
* Processing data - Kaitlyn  
* Vectorize DNA Sequences** - Gen and Kaitlyn 
    * Kaitlyn: k-mers, natural (and sub-sequence) vectors
    * Gen: Word2Vec, other natural language encoding methods 
* Feature Selection: Alana 
* Machine Learning and Deep Learning: Kyle - Done!
* Multi-task Learning - ? 

##### Paper
* Background and introduction: Kaitlyn 
* Methods 
    * Data-processing: Kaitlyn 
    * Sequence vectorization: Kaitlyn and Gen 
    * Feature Selection: Alana 
    * ML/DL: Kyle 
* Results and Discussion: Kyle 
* Figures: Kyle  

#### Dataset and Availability 

##### Dataset 
* `hiv-db.tar.xz`: contains 20,386 unprocessed sequences from 289 subtypes [LANL HIV Sequence Database](https://www.hiv.lanl.gov/components/sequence/HIV/search/search.html)
* After processing there are 15,018 sequences and 28 subtypes. 
* Each sequence is approx. 10,000 nucleotides.
* `hiv.txt`: the sequences (atgcgctagatcga) 
    * Each line corresponds to one sequence
* `labels.txt`: the subtype label 
    * The first line in `labels.txt` corresponds to the first sequence (line) in `hiv.txt`. 

##### Scripts 
* `pre-process.py`: reads `hiv-db.tar.xz`
    * Removes sequences with unknown characters (eg: N) and subtypes with too few examples 
    * Creates `hiv.txt` and `labels.txt`

`hiv.txt` can be used to vectorize the sequences.  

* `classify.py`: take X and y as inputs, where X is a 2D feature vector and Y is a vector
   * Before runing the codes, please first set configs, including output directory, possible hyper-parameters (optional)
   * Output: Confusion metrices and numerical results, which will be saved in your output directory.

* `feature_selection.ipynb`: script for dimensionality reduction (PCA, LR-Lasso, 1D-CNN). Specific instructions can be found in the ipython notebook.
