#date: 2023-11-14T17:06:16Z
#url: https://api.github.com/gists/17ebc6ed7f4ef0537f6672de770ab9cf
#owner: https://api.github.com/users/xelicrojas


if env.user.id not in [
  590 #Vivian
  ]: 
  raise UserError ('Usted no está autorizado para ejecutar esta acción')

guara_obj = 'sbd.guarantee.track'
batch_guarantee_model = 'sbd.batch.guarantee'
batch_guarantee_id = 626
message_body = 'Ticket: Error en carga Credecoop (#24157). Estado de validación modificado debido a que se cargó la salida de la garantía tiempo antes. ID de seguimiento de garantía: %s.'

ids = [42338, 42408, 42442]

for guarantee_id in ids:
    guarantee = env[guara_obj].browse(guarantee_id)
    if not guarantee:
        log('Garantía no encontrada: %s' % (guarantee_id))
        continue
    try:
        guarantee.write({
            'validity_status': 'active',
        })

        batch_guarantee = env[batch_guarantee_model].browse(batch_guarantee_id)
        batch_guarantee.message_post(body=message_body % guarantee_id)

        log('Garantía modificada %s' % (guarantee_id))
        env.cr.commit()
    except Exception as e:
        log('Garantía falló %s' % repr((guarantee_id, e)))
        env.cr.rollback()