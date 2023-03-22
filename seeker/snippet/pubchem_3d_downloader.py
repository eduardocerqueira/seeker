#date: 2023-03-22T17:03:40Z
#url: https://api.github.com/gists/7c4da756238fdf02cadcd84ca18c4bc0
#owner: https://api.github.com/users/FarisIzzaturRahman

def download_sdf(cid: int) -> str:
    """Download the 3D structure of a compound in SDF format from PubChem and return the SDF text."""
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF'
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except (requests.exceptions.RequestException, pcp.PubChemHTTPError) as e:
        print(f"Error downloading SDF for CID {cid}: {e}")
        return None