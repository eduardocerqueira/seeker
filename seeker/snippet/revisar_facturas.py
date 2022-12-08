#date: 2022-12-08T16:53:19Z
#url: https://api.github.com/gists/67ee6363fd8a960ad32d8a89d2bcd967
#owner: https://api.github.com/users/edy1192

invoices = env["account.move"].search([
  ("state", "=", "posted"), ("move_type", "in", ["in_invoice", "out_invoice"]), ("commission_type", "in", ["commission", "recovery"]), ("origin_budget_move_id", "!=", False), ("budget_move_id", "!=", False)])

invoices = invoices.filtered(lambda i: i.origin_budget_move_id.is_reversed or i.budget_move_id.is_reversed)

domain = [('id', 'in', invoices.ids)]

act = {
  'name': 'Facturas',
  'type': 'ir.actions.act_window',
  'res_model': 'account.move',
  'target': 'current',
  'view_mode': 'tree,form'}
act['domain'] = domain
action = act