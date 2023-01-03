#date: 2023-01-03T16:48:32Z
#url: https://api.github.com/gists/b620c22ce2f31e698622508b64d791e0
#owner: https://api.github.com/users/hugho-ad

attach = env['ir.attachment'].search([
    ('name', '=', 'products_do_merge.csv')])
import_wizard = env['base_import.import'].create({
    'res_model': 'product.product',
    'file': attach.index_content,
      'file_type': 'text/csv'})    
data = import_wizard._convert_import_data(['new_id', 'old_id'], {'quoting': '"', 'separator': ',', 'headers': True})[0]
