#date: 2023-03-22T17:00:50Z
#url: https://api.github.com/gists/b4ba5147dace33bf90d1ea0042cd8da5
#owner: https://api.github.com/users/FarisIzzaturRahman

def search_pubchem(compound: str) -> int:
    """Search PubChem for a given compound and return its CID."""
    try:
        result = pcp.get_compounds(compound, 'name', record_type='3d')[0]
        return result.cid
    except (IndexError, pcp.PubChemHTTPError) as e:
        print(f"Error searching PubChem for {compound}: {e}")
        return None