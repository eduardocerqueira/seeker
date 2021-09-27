#date: 2021-09-27T17:07:16Z
#url: https://api.github.com/gists/adc733fa2168fda39efa36f0f104fab4
#owner: https://api.github.com/users/dnicolodi

class EntryPrinter(printer.EntryPrinter):
    def __init__(self, currency_column=50):
        super().__init__()
        self.target_currency_column = currency_column

    def Transaction(self, entry, oss):
        # Compute the string for the payee and narration line.
        strings = []
        if entry.payee:
            strings.append('"{}"'.format(misc_utils.escape_string(entry.payee)))
        if entry.narration:
            strings.append('"{}"'.format(misc_utils.escape_string(entry.narration)))
        elif entry.payee:
            # Ensure we append an empty string for narration if we have a payee.
            strings.append('""')

        oss.write('{e.date} {e.flag} {}\n'.format(' '.join(strings), e=entry))

        if entry.tags:            
            for tag in sorted(entry.tags):
                print(f'#{tag}', file=oss)
        if entry.links:
            for link in sorted(entry.links):
                print(f'^{link}', file=oss)
        
        self.write_metadata(entry.meta, oss)

        rows = [self.render_posting_strings(posting) for posting in entry.postings]
        strs_account = [row[0] for row in rows]
        strs_position, width_position = printer.align_position_strings(row[1] for row in rows)
        strs_weight, width_weight = printer.align_position_strings(row[2] for row in rows)

        width_number = re.search(r'[A-Z]', strs_position[0]).start() - 1
        width_account = self.target_currency_column - width_number

        if self.render_weight and any(map(has_nontrivial_balance, entry.postings)):
            fmt = "{0}{{:{1}}}  {{:{2}}}  ; {{:{3}}}\n".format(
                self.prefix, width_account, width_position, width_weight).format
            for posting, account, position, weight in zip(entry.postings,
                                                          strs_account,
                                                          strs_position,
                                                          strs_weight):
                oss.write(fmt(account, position, weight))
                if posting.meta:
                    self.write_metadata(posting.meta, oss, '    ')
        else:
            fmt_str = "{0}{{:{1}}}  {{:{2}}}".format(
                self.prefix, width_account, max(1, width_position))
            fmt = fmt_str.format
            for posting, account, position in zip(entry.postings, strs_account, strs_position):
                print(fmt(account, position).rstrip(), file=oss)
                if posting.meta:
                    self.write_metadata(posting.meta, oss, '    ')
