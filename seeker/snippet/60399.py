#date: 2023-04-04T17:05:10Z
#url: https://api.github.com/gists/b92791565e99fc9e4ac53f866f5377fe
#owner: https://api.github.com/users/pablohmontenegro

# aplicar debajo de esta línea https://github.com/odoo/odoo/blob/13.0/addons/account/models/reconciliation_widget.py#L231
# Esto es para tratar de que no me proponga conciliar líneas antiguas
from datetime import datetime
from dateutil.relativedelta import relativedelta
date_from = fields.Date.today() - relativedelta(months=3)
bank_statement_line_ids_filtered = self.env['account.bank.statement.line'].search([('id', 'in', bank_statement_line_ids), ('date', '>', date_from)]).mapped('id')
bank_statement_line_ids = bank_statement_line_ids_filtered