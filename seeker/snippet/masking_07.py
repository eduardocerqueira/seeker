#date: 2023-07-04T16:44:00Z
#url: https://api.github.com/gists/4b7d6403e5d25523a0e94f6e018ead54
#owner: https://api.github.com/users/aarroonn22

from transformers import default_data_collator


def custom_collator(
    features: list,
    tokeniser,
):
    max_len = sum(
        max(
            [features[i]['attention_mask'] for i in range(len(features))]
        )
    )
    
    for feature in features:
        pad_feature(
            input_list=feature['input_ids'],
            max_len=max_len,
            padding_val= "**********"
        )
        pad_feature(
            input_list=feature['attention_mask'],
            max_len=max_len,
            padding_val= "**********"
        )
        pad_feature(
            input_list=feature['labels'],
            max_len=max_len,
            padding_val=-100,
        )
    
    return default_data_collator(features)
_collator(features)
