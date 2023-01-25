Before Running the above Codes, please install the following Packages by using the conda command line:

(1) Pandas (conda install -c conda-forge pandas)
(2) Vaex (conda install -c conda-forge vaex)
(3) Pyspark (conda install -c conda-forge pyspark)
(4) Tensorflow (pip install tensorflow)
(5) Keras (pip install keras)
(6) Pytorch (conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.2 -c pytorch)
(7) CuPy 9.0 (conda install -c conda-forge cupy=9.0)
(8) CUDA 11.2 (conda install -c anaconda cudatoolkit=11.2)
(9) Tensorflow-gpu (pip install tensorflow-gpu)
(10) Spacy (conda install -c conda-forge spacy)
(11) Spacy English Model (python -m spacy download en_core_web_trf)
(12) Numpy (conda install -c conda-forge numpy)
(13) Seaborn (conda install -c conda-forge seaborn)
(14) Matplotlib (conda install -c conda-forge matplotlib)
(15) Scikit Learn (conda install -c conda-forge scikit-learn)
(16) Kneed (conda install -c conda-forge kneed)
(17) Tqdm (conda install -c conda-forge tqdm)
(18) Numba (conda install -c conda-forge Numba) 
(19) Gensim (conda install -c conda-forge gensim)
(20) Tabulate (conda install -c conda-forge tabulate)
(21) Scipy (conda install -c conda-forge scipy)
(22) NLTK (conda install -c conda-forge nltk)

INSTRUCTIONS TO RUN THE CODE:

The Code is divided into 4 parts because the RAM of the System was limited (16 GB) due to which loading saved data caused MemorryError when loading massive files in just one file. 
So if we execute one model after the other then the Kernel of Jupyter Restarts due to which all the data is lost.

IMPORTANT: Please, shutdown and close each .ipynb File after execution to reset the Kernel.


(1) First, execute the "Assignment 2 Data Preparation Part 1.ipynb" File, 
which will Preprocess the Data and will save it to the D drive or any other desired location of the Computer.

(2) After this, execute the "Assignment 2 Data Preparation Part 2.ipynb" File, 
which will Preprocess the Data and will save it to the D drive or any other desired location of the Computer.

(3) Then, execute "Assignment 2 Doc2Vec Model.ipynb" File to run the Doc2Vec Model with Question, Mean and Concatenated Sentence Embedding to generate Relevance Score.

(4) Next, execute "Assignment 2 Word2Vec Model.ipynb" File to run the Word2Vec Model with Question, Mean and Concatenated Sentence Embedding to generate Relevance Score.