#date: 2022-11-17T17:14:47Z
#url: https://api.github.com/gists/d68cb766cfbf37bb1fcb01fe125f519a
#owner: https://api.github.com/users/ofelix03

def action_send_email_interested(self):

    additional_values = {
        
    }
    email_values = {
        "email_to": None,
        "email_from": None,
    }
    mail_template = self.env.ref('realtor.interested_property_template')
    mail_template.with_context(additional_values).send_mail(
        self.id, email_values=email_values
    )
