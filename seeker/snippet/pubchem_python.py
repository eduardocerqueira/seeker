#date: 2023-03-22T17:05:04Z
#url: https://api.github.com/gists/09f4096eb44bd9dc3597fbe20cf140bf
#owner: https://api.github.com/users/FarisIzzaturRahman

def main(csv_file: str) -> None:
    """Main function that reads a CSV file of compound names, 
    searches PubChem for each compound, and downloads the 3D
    structure of each compound in SDF format."""
    # Load table data into Pandas DataFrame
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Loop through DataFrame and search PubChem for each compound
    cids = []
    for index, row in df.iterrows():
        compound_name = row['compound_name']
        cid = search_pubchem(compound_name)
        cids.append(cid)
    df['cid'] = cids

    # Loop through DataFrame and download the 3D structure of each compound
    for index, row in df.iterrows():
        compound_name = row['compound_name']
        cid = row['cid']
        if cid is None:
            print(f"Error: No CID found for {compound_name}")
            continue

        sdf = download_sdf(cid)
        if sdf is None:
            print(f"Error: Unable to download SDF for {compound_name}")
            continue

        try:
            with open(f'{compound_name}.sdf', 'w') as f:
                f.write(sdf)
            print(f"Downloaded SDF for {compound_name}")
        except IOError as e:
            print(f"Error saving SDF for {compound_name}: {e}")