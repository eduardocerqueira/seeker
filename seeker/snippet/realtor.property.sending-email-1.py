#date: 2022-11-17T17:13:34Z
#url: https://api.github.com/gists/0d2866269cfc6578edd26369941ccaf9
#owner: https://api.github.com/users/ofelix03

def action_send_email_interested(self):
    template = self.env.ref('realtor.interested_property_template')
    template.body_html.replace('--property-name--', self.name)
    template.body_html.replace('--property-manager-name--', self.manger_id.name)
    template.body_html.replace('--customer-name--', self.env.user.id)
    mail_values = {
        'email_to': None,
        'email_from': None,
        'subject': None,
        'body_html': template.body_html,
        'message_type': email
    }
    self.env["mail.mail"].create(mail_values).send()