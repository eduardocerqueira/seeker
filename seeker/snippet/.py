#date: 2024-01-02T17:04:06Z
#url: https://api.github.com/gists/a32953fed003c61705cc57db02b17b44
#owner: https://api.github.com/users/DIOR27

    def attachmentReview(self):
        policies = self.env["issue.policy"].search([])

        for policy in policies:
            attachments = self.env["ir.attachment"].search(
                [
                    ("res_model", "=", "issue.policy"),
                    ("res_id", "=", policy.id)
                ]
            )

            count = len(attachments) or None
            if count:
                attachment_names = "\n, ".join(attach.name for attach in attachments)

            policy.message_post(
                body="El %s tiene %s adjuntos:\n\n%s" % (policy.name, count, attachment_names),
            )
            
            self.env.cr.commit()