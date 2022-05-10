#date: 2022-05-10T17:02:04Z
#url: https://api.github.com/gists/d7925e8aac35473e9e6ae10f84ed20a9
#owner: https://api.github.com/users/AlessandroMondin

def cross_entropy(self, scaled_logits, one_hot):
    if self.library == "tf":
        masked_logits = tf.boolean_mask(scaled_logits, one_hot)
        ce = -tf.math.log(masked_logits)
    else:
        masked_logits = torch.masked_select(scaled_logits, one_hot)
        ce = -torch.log(masked_logits)
    return ce