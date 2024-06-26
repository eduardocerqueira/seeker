#date: 2024-06-26T16:38:51Z
#url: https://api.github.com/gists/d67f76987b7f392daab76ef4c6c0c037
#owner: https://api.github.com/users/DotTry

def save_and_upload_model():
    PATH = 'model.pth'

    with io.FileIO(PATH, 'wb') as file:
        torch.save(model, file)

    with io.FileIO(PATH, 'rb') as file:
        model = torch.load(file)
        
    # Hard coded until we link service layer
    campaignId = 'afd62e6c-6e18-4255-b9ba-858ee86c9598'

    try:
        targetHost="127.0.0.1"  # Target database host
        targetUser="root"  # Target database user
        targetPassword= "**********"
        targetDatabase="db"  # Target database name

        target_conn = mysql.connector.connect(
            host=targetHost,  # Target database host
            user=targetUser,  # Target database user
            password= "**********"
            database=targetDatabase  # Target database name
        )
        
        with open(PATH, 'rb') as file:  # Replace 'path_to_your_file' with the actual file path
            model_data = file.read()
            sql = "INSERT INTO leadScoreModels  (namespaceId, campaignId, id, model, version, createdAt, updatedAt, deletedAt)  VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            val = (namespaceId, campaignId, '123', model_data, '1', datetime.datetime.now(), datetime.datetime.now(), None) 

            with target_conn:
                with target_conn.cursor() as cursor:
                    cursor.execute(sql, val)
                    target_conn.commit()
                    print("Model uploaded successfully!")

    except Error as e:
        print(f"Error uploading model: {e}")

    finally:
        if target_conn.is_connected():
            target_conn.close()

save_and_upload_model();          target_conn.close()

save_and_upload_model();