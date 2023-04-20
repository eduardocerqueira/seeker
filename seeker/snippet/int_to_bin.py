#date: 2023-04-20T17:00:20Z
#url: https://api.github.com/gists/2ce31447b205a53baf692ba94476491f
#owner: https://api.github.com/users/Goatghosts

def int_to_binary_with_fixed_length(n, length):
    binary_repr = format(n, f"0{length}b")
    return binary_repr


def modify_binary(binary_repr):
    new_binary_repr = "1" + binary_repr[:-1]
    return new_binary_repr


def binary_to_int(binary_repr):
    return int(binary_repr, 2)


def main():
    number = 1545353432434673534542345234
    fixed_length = 256
    binary_repr = int_to_binary_with_fixed_length(number, fixed_length)
    modified_binary_repr = modify_binary(binary_repr)
    result = binary_to_int(modified_binary_repr)
    print(f"Original number: {number}")
    print(f"Binary representation with fixed length: {binary_repr}")
    print(f"Modified binary representation: {modified_binary_repr}")
    print(f"Resulting integer: {result}")


if __name__ == "__main__":
    main()
