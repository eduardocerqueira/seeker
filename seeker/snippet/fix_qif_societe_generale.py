#date: 2023-04-10T16:44:00Z
#url: https://api.github.com/gists/0f3055ab04eef81153c48795614d89f6
#owner: https://api.github.com/users/Uinelj

if __name__ == "__main__":
    fp = "foo.qif"

    # open with correct encoding
    with open(fp, "r", encoding="iso-8859-1") as f:
        lines = list(f)

        # fix !type:Bank to !Type:Bank
        if lines[0].startswith("!type"):
            lines[0] = lines[0].replace("t", "T", 1)

            # overwrite (utf-8)
            with open(fp, "w", encoding="utf-8") as f:
                f.writelines(lines)
        else:
            print("nothing to do")