#date: 2021-12-17T17:08:26Z
#url: https://api.github.com/gists/b6cf08541a469d5651df5194dd2afa80
#owner: https://api.github.com/users/michael-aloys

# Just a scrap code to try out the z-multiplication for sequence labeling
import torch


def main():
    # dataset with 2 instances
    dataset_x = [["Vienna", "is", "in", "Austria"], ["Anne", "met", "Bruno", "in", "Cologne"]]
    dataset_y_true = [["LOC", "O", "O", "LOC"], ["PER", "O", "PER", "O", "LOC"]]

    label_to_label_idx = {"O":0, "LOC":1, "PER":2}

    # z matrix for each instance (#tokens x #rules)
    # with two rules, one for location, one for person
    # and both rules are perfect (i.e. they match the ground truth labels)
    # instance 1
            #LOC-rule  PER-rule
    z_matrix_1 = [[1, 0], # Vienna
                  [0, 0], # is
                  [0, 0], # in
                  [1, 0]] # Austria

    z_matrix_2 = [[0, 1], # Anne
                  [0, 0], # met
                  [0, 1], # Bruno
                  [0, 0], # in
                  [1, 0]] # Cologne

    # rules-labels matrix (#rules x #labels)
                 #O  LOC  PER
    rl_matrix = [[0, 1, 0], #LOC-rule
                 [0, 0, 1]] # PER-rule

    z_matrix_1 = torch.Tensor(z_matrix_1)
    z_matrix_2 = torch.Tensor(z_matrix_2)
    rl_matrix = torch.Tensor(rl_matrix)

    # multiplying z_matrix_1 * rl_matrix should give us a matrix (#tokens x #labels)
    # which we could feed into a classifier
    print(f"Tensor tokens x labels for instance 1:\n{torch.matmul(z_matrix_1, rl_matrix)}\n")
    print(f"Tensor tokens x labels for instance 2:\n{torch.matmul(z_matrix_2, rl_matrix)}\n\n")

    # stacking the z_matrices into one tensor, we can join all instances:
    # (#instances x #tokens x #rules)
    # The z-matrices have different length. So we have to add padding to put them all in one Tensor
    z_tensor = torch.nn.utils.rnn.pad_sequence([z_matrix_1, z_matrix_2],
                                                 batch_first=True, # the "batch" here is all instances
                                                 padding_value=-1)
    print(f"Stacked and padded z-matrices:\n {z_tensor}\n\n")

    # multiplying that tensor with the rl_matrix gives us
    # (#instances x #tokens x #labels)
    # which is the representation that Huggingface Transformers usually uses afaik
    print(f"Tensor tokens x labels for all instances:\n{torch.matmul(z_tensor, rl_matrix)}\n")

    # Issue: The padded entry of the first instance has not only -1 entries but also a 0 (due to the multiplication)
    # [ 0., -1., -1.]
    # One has to then be careful to use a correct mask (which Huggingface Transformer anyways requires for many models)
    # so that the models does not think this is  true label.
    # Or we could provide a function that does the multiply and corrects the padding.

    # For the pre-processing and storing, my suggestion would be to store the z-matrices as list of z-matrices (and
    # not as Tensors). And then let the Trainer decide what to do with them. And offer a utility function that
    # converts list of z-matrices into a z-tensor (so bascially line 45) and one for z-tensor * rl_matrix with fixing
    # padding. Just a suggestion, happy to discuss this.

if __name__ == "__main__":
    main()
