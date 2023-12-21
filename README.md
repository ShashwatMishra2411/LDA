# Topic Modeling from PDF Documents

## Overview

This Python script performs topic modeling on a collection of PDF documents using Latent Dirichlet Allocation (LDA) and visualizes the results. The script extracts text from PDF files, preprocesses the data, trains an LDA model, and generates visualizations to explore document-topic relationships.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python (>=3.6)
- Required Python packages: pdfplumber, numpy, glob, seaborn, matplotlib, gensim, spacy

## Project Structure

- `Articles/`: Directory containing PDF files to be processed.
- `gist_stopwords.txt`: File containing additional stopwords.
- `lda_topic_modeling.py`: The main Python script for topic modeling.

## Usage

1. Ensure all prerequisites are installed.
2. Place PDF files in the `Articles/` directory.
3. Run the script using the command: `python lda_topic_modeling.py`

## Output

The script produces the following outputs:

- **Document-Topic Matrix Heatmap:** Visual representation of the probability distribution of topics for each document. 
![HeatMap](https://github.com/ShashwatMishra2411/LDA/assets/134842493/501554b6-c1e0-4214-98cc-550c1b1b4a26)
- **Topic Word Distribution Plots:** Bar plots showing the probability distribution of words in each topic.
![BarGraph](https://github.com/ShashwatMishra2411/LDA/assets/134842493/528b648d-b797-4885-8c27-058dd4af0028)

## Additional Notes

- The script uses spaCy for lemmatization and the en_core_web_sm model. Make sure it is installed (`python -m spacy download en_core_web_sm`).
- The stopwords can be customized by modifying the `gist_stopwords.txt` file.

## Author

Shashwat Mishra(22BCE1853)\
Dhruv Chauhan(22BCE1837)
