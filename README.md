# GPT-InvestAR

Enhancing Stock Investment Strategies through Annual Report Analysis with Large Language Models

This repository contains a set of tools and scripts designed to enhance stock investment strategies through the analysis of annual reports using Large Language Models. The components in this repository are organized as follows:

1. **download_10k.py**: This Python script downloads 10-K filings of companies from the SEC website, which contain crucial financial information.

2. **convert_html_to_pdf.py**: Converts HTML files to PDF files. PDFs are preferred due to their token efficiency for further analysis.

3. **make_targets.py**: Generates a DataFrame of stock tickers with target values of different time resolutions, which can be used as investment targets for a Machine Learning model.

4. **embeddings_save.py**: Generates embeddings of PDF files and saves them using Cromadb. These embeddings are numerical representations of the textual content in annual reports.

5. **gpt_scores_as_features.py**: Utilizes saved embeddings to query all questions for each annual report using a Large Language Model (LLM) such as GPT-3.5, and uses the scores or answers as features.

6. **modeling_and_return_estimation.ipynb**: This Jupyter Notebook contains the core modeling process. It uses machine learning techniques, specifically Linear Regression, to model the dataset and estimate returns. The goal is to create a portfolio of top-k predicted stocks and compare their returns with the S&P 500 index.

By following the sequence of these components, you can analyze annual reports, generate embeddings, and build predictive models to potentially enhance stock investment strategies.

Feel free to explore each component for more details and usage instructions.


## Dependencies

1. [LLama Index](https://github.com/jerryjliu/llama_index) (and related dependencies)

2. [OpenBB](https://github.com/OpenBB-finance/OpenBBTerminal) (and related dependencies)

3. [Scikit-Learn](https://github.com/scikit-learn/scikit-learn)

4. [PDFKit](https://github.com/JazzCore/python-pdfkit) (and related dependencies)

It is recommended to install libraries 1 and 2 in separate virtual (conda) environments. The python scripts mentioned above do not require both these libraries to be installed in the same environment.

