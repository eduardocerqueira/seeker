#date: 2024-03-12T16:47:43Z
#url: https://api.github.com/gists/22442be3b9b1a558b4d254570b9373f1
#owner: https://api.github.com/users/mem-adhoc

errors = []
for company in env['res.company'].search([('partner_id.country_id.code', 'in', ['AR']), ('reconcile_on_company_currency', '=', True)]):
  moves = env['account.move'].search([('company_id', '=', company.id), ('state', '=', 'posted'), ('amount_total_in_currency_signed', '!=', 0.0), ('currency_id', '!=', company.currency_id.id), ('create_date', '>=', '2023-04-04'), ('journal_id.type', 'in', ['sale', 'purchase'])])
  datos_correcciones = {}
  for move in moves:
    # ['amount_total_in_currency_signed', 'amount_total_signed', rate_field, 'invoice_date', 'move_type', 'name', 'amount_residual_signed', 'payment_state'])
    correct_amount = (move.amount_total_in_currency_signed * move.computed_currency_rate)
    if abs(move.amount_total_signed - correct_amount) < 100:
      continue

    msg = 'Arreglando (%s - %s), Importe: Correcto %s, Actual %s' % (move.id, move.name, correct_amount, move.amount_total_signed)
    datos_correcciones.append({
        move.id: move.payment_group_ids
    })
    move = move.with_context(check_move_validity=False)
    move.button_draft()
    rate = move.computed_currency_rate
    context = {
      'tax_list_origin': move.mapped('invoice_line_ids.tax_ids'),
      'tax_total_origin': move.tax_totals
    }
    move.with_context(context).l10n_ar_currency_rate = 2.0
    move.with_context(context).l10n_ar_currency_rate = rate

    # hacemos mayor a $1 por temas de redondeo
    if abs(move.amount_total_signed - correct_amount) > 10:
      errors.append(move.id)
      # env.cr.rollback()

    # if move._is_dummy_afip_validation():
    #     move.l10n_ar_afip_auth_code = False
  for move in datos_correcciones():
    #move va a ser un diccionario del tipo {move.ids : move.payment_group_ids}
    #acareconcilias nuevamente cada factura con sus pagos correspondientes
    move.action_post()
    print (msg + ' Nuevo importe %s, Diff %s' % (move.amount_total_signed, correct_amount - move.amount_total_signed))
    env.cr.commit()

print("errors %s" % errors)