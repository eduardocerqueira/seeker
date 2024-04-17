#date: 2024-04-17T17:02:03Z
#url: https://api.github.com/gists/af0236b2067dad3e4460ba1d3b2fa43e
#owner: https://api.github.com/users/hugoleborso

def main():
    parser = argparse.ArgumentParser(
        description="Compute embeddings for images using Vertex AI."
    )
    parser.add_argument(
        "dataset_folder", type=str, help="The folder containing the dataset images."
    )
    args = parser.parse_args()
    dataset_folder = args.dataset_folder
    vector_db = VectorDB()
    print(f"Computing embeddings for dataset in folder: {dataset_folder}")

    os.makedirs("backend/engine-imgs", exist_ok=True)
    for file in Path(dataset_folder).rglob("*"):
        if file.suffix in ALLOWED_FILE_EXTENSIONS:
            shutil.copy2(file, "backend/engine-imgs")

    embeddings, images_path = load_image_embeddings(dataset_folder)
    print(f"Inserting {len(embeddings)} embeddings into the database.")
    for i, embedding in enumerate(embeddings):
        vector_db.insert(i, embedding, {"image_path": str(images_path[i].name)})
    return {"message": "Embeddings computed and inserted into the database."}
    

if __name__ == "__main__":
    main()