#date: 2025-11-26T16:56:45Z
#url: https://api.github.com/gists/e8d1b1c30b69fb6d0579787a6ba27068
#owner: https://api.github.com/users/sybdeb

# -*- coding: utf-8 -*-

from odoo import models, fields

class ProductTemplate(models.Model):
    """Extend product.template with UNSPSC field"""
    _inherit = 'product.template'
    
    # Extra product classification field
    unspsc = fields.Char('UNSPSC', help="United Nations Standard Products and Services Code")

class ProductProduct(models.Model):
    """Extend product.product with UNSPSC field"""
    _inherit = 'product.product'
    
    # Make field available on product variant level too
    unspsc = fields.Char(related='product_tmpl_id.unspsc', string='UNSPSC', readonly=False, store=True)