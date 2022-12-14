#date: 2022-12-14T16:56:44Z
#url: https://api.github.com/gists/ca0a62c4861b1e6f0be27231ba8e6e7f
#owner: https://api.github.com/users/rustyguts

b2 update-bucket --corsRules '[
    {
        "corsRuleName": "downloadFromAnyOriginWithUpload",
        "allowedOrigins": [
            "*"
        ],
        "allowedHeaders": [
            "*"
        ],
        "allowedOperations": [
            "b2_download_file_by_id",
            "b2_download_file_by_name",
            "b2_upload_file",
            "b2_upload_part"
        ],
        "maxAgeSeconds": 3600
    }
]' my-awesome-bucket-name allPublic