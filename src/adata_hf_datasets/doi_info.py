import requests
import re
from bs4 import BeautifulSoup

import pandas as pd
import anndata

def map_dataset_to_doi(census_datasets: pd.DataFrame, adata: anndata.AnnData) -> anndata.AnnData:
    """
    Maps dataset IDs in an AnnData object to their corresponding collection DOIs using the census datasets DataFrame.
    This works only for adata obtained from the cellxgene census.

    Parameters
    ----------
    census_datasets : pd.DataFrame
        A DataFrame containing dataset metadata, including "dataset_id" and "collection_doi".
    adata : anndata.AnnData
        An AnnData object with a column "dataset_id" in adata.obs.

    Returns
    -------
    anndata.AnnData
        The updated AnnData object with a new column "collection_doi" in adata.obs.
    """
    # Ensure dataset_id exists in adata.obs
    if "dataset_id" not in adata.obs:
        raise ValueError("The 'dataset_id' column is missing from adata.obs.")

    # Get unique dataset IDs from adata
    unique_dataset_ids = adata.obs["dataset_id"].unique()

    # Filter census_datasets for relevant dataset IDs
    filtered_census = census_datasets[census_datasets["dataset_id"].isin(unique_dataset_ids)]

    # Build a mapping dictionary {dataset_id: collection_doi}, handling missing DOIs
    dataset_to_doi = dict(zip(filtered_census["dataset_id"], filtered_census["collection_doi"].fillna("No DOI")))

    # Map the collection DOI to adata.obs["collection_doi"]
    return dataset_to_doi

class DOIFetcher:
    """A class to fetch metadata and full-text of a research paper using its DOI from Europe PMC."""

    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def fetch_pmc_info(self, doi):
        """
        Fetches metadata for a paper using its DOI from Europe PMC.

        Parameters
        ----------
        doi : str
            The DOI of the publication.

        Returns
        -------
        dict or str
            A dictionary containing metadata if found, otherwise an error message.
        """
        url = f"{self.base_url}/search?query={doi}&format=json"
        response = requests.get(url)

        if response.status_code != 200:
            return "Error: Unable to fetch metadata"

        data = response.json()
        results = data.get("resultList", {}).get("result", [])

        # Find exact DOI match
        for result in results:
            if result.get("doi", "").lower() == doi.lower():
                # Ensure the paper is open-access before returning
                if result.get("isOpenAccess") == "Y":
                    return result
                else:
                    return "Error: Paper is not open access"

        return "Error: No results found in Europe PMC."

    def get_full_paper(self, doi):
        """
        Fetches the full text of an open-access paper using its PMC ID.

        Parameters
        ----------
        doi : str
            The DOI of the paper.

        Returns
        -------
        str
            The full-text content if retrievable, otherwise an error message.
        """
        paper_info = self.fetch_pmc_info(doi)

        if isinstance(paper_info, str):  # Error message
            return paper_info

        pmcid = paper_info.get("fullTextIdList", {}).get("fullTextId", [None])[0]
        if not pmcid:
            return "Error: No PMC ID found for full text retrieval."

        # Retrieve full text from PMC
        return self.fetch_full_text_from_pmc(pmcid)

    def fetch_full_text_from_pmc(self, pmcid):
        """
        Retrieves the full text of an open-access paper using its PMC ID.

        Parameters
        ----------
        pmcid : str
            The PMC ID of the paper.

        Returns
        -------
        str
            The cleaned full-text content.
        """
        url = f"{self.base_url}/{pmcid}/fullTextXML"
        response = requests.get(url)

        if response.status_code != 200:
            return "Error: Unable to fetch full text."

        soup = BeautifulSoup(response.content, "html.parser")
        full_text = soup.get_text(separator="\n")
        return self.clean_full_text(full_text)

    def clean_full_text(self, text):
        """
        Cleans extracted full-text by removing excessive newlines and unnecessary spaces.

        Parameters
        ----------
        text : str
            The raw extracted text.

        Returns
        -------
        str
            The cleaned and formatted text.
        """
        text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
        text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spaces
        text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)  # Join words broken across lines
        text = re.sub(r'(\.|\;|\:)\n', r'\1 ', text)  # Ensure punctuation spacing
        return text.strip()

    def extract_section(self, text, section_name):
        """
        Extracts a specific section (e.g., Abstract, Methods) from full-text.

        Parameters
        ----------
        text : str
            The full-text content of the paper.
        section_name : str
            The section name to extract (e.g., "Abstract", "Methods").

        Returns
        -------
        str
            Extracted section text or an error message.
        """
        pattern = rf"{section_name}\s*(.*?)\n[A-Z]"  # Match section text until the next capitalized header
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()
        return f"Error: {section_name} section not found."