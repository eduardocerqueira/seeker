#date: 2022-04-21T17:12:10Z
#url: https://api.github.com/gists/430c226cfe98ec3ed9dbc2de88b05388
#owner: https://api.github.com/users/JulioSerna

sm_obj = env["stock.move"]

quants = env["stock.quant"].search([]).filtered(lambda q: q.location_id.usage == "internal")
for quant in quants:
  rounding = quant.product_id.uom_id.rounding
  
  ## Incoming ##
  domain = [
    ("product_id", "=", quant.product_id.id),
    ("state", "=", "done"),
    ("location_dest_id", "=", quant.location_id.id),
  ]
  moves = sm_obj.search(domain)
  incoming = sum(moves.mapped("product_uom_qty"))

  ## Outgoing ##
  domain = [
    ("product_id", "=", quant.product_id.id),
    ("state", "=", "done"),
    ("location_id", "=", quant.location_id.id),
  ]
  moves = sm_obj.search(domain)
  outgoing = sum(moves.mapped("product_uom_qty"))

  real_quantity = incoming - outgoing
  diff = real_quantity - quant.quantity

  if float_compare(diff, 0.0, precision_rounding=rounding) != 0:
    original = quant.quantity
    quant._update_available_quantity(quant.product_id, quant.location_id, diff, lot_id=quant.lot_id, package_id=quant.package_id, owner_id=quant.owner_id)
    msg = """Updating %s | %s\n %s:\n  Original Quant Qty:%f\n  Diff with Moves:%f\n  Qty After Update:%f\n\n""" % (
      quant.product_id, quant.product_id.name, quant, original, diff, quant.quantity)
    log(msg)

#raise Warning(msg)