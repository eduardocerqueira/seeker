#date: 2021-12-28T16:43:31Z
#url: https://api.github.com/gists/7a1f1400bab57e399e43cc8f7db37bdc
#owner: https://api.github.com/users/MarJene

from tqdm import tqdm

pol_o = c.GiscedataPolissa
f1_obj = c.model('giscedata.facturacio.importacio.linia')

f1_import_phase_10 = f1_obj.search([('cups_id','=',False),('type_factura','=','R')])
cups_f1_import_phase_10 = f1_obj.read(f1_import_phase_10, ['cups_text'])

pols_endarr = pol_o.search([('facturacio_endarrerida', '=', True), ('tarifa.name','ilike','2.%')],context={'active_test':False})
pols_f1_r = []
for p_id in tqdm(pols_endarr):                                                                                                          
   p_info = pol_o.read(p_id, ['cups','data_alta','name'])
   f1_r = f1_obj.search([('cups_id','=',p_info['cups'][0]),('fecha_factura_desde','>=',p_info['data_alta']),('type_factura','=','R')])
   if len(f1_r) > 0: 
       pols_f1_r.append(p_info['name'])

with open('/tmp/polisses_2_with_F1R.csv','w') as f:             
    f.write("\n".join([str(p) for p in list(set(pols_f1_r))]))  

