#date: 2025-11-26T16:56:45Z
#url: https://api.github.com/gists/e8d1b1c30b69fb6d0579787a6ba27068
#owner: https://api.github.com/users/sybdeb

# -*- coding: utf-8 -*-

from odoo import models, fields

class ProductSupplierinfo(models.Model):
    """Extend product.supplierinfo with extra supplier fields"""
    _inherit = 'product.supplierinfo'
    
    # Override existing field label
    price = fields.Float('Ink.Prijs', help="Purchase price from this supplier")
    
    # Extra supplier fields voor CSV import
    order_qty = fields.Float('Bestel Aantal', default=0.0, help="Minimum order quantity from supplier")
    supplier_stock = fields.Float('Voorraad Lev.', default=0.0, help="Current stock at supplier")  
    supplier_sku = fields.Char('Art.nr Lev.', help="Supplier's internal SKU/article number")
    
    # Product identification fields - inherited from product voor Smart Import matching
    product_name = fields.Char('Product Naam', 
                              related='product_tmpl_id.name', 
                              readonly=True,
                              help="Product name from product for CSV matching/reference")
    product_barcode = fields.Char('Product EAN/Barcode', 
                                 related='product_id.barcode',
                                 readonly=True,
                                 help="EAN/Barcode from product for CSV matching")
    product_default_code = fields.Char('Product SKU/Ref', 
                                      related='product_tmpl_id.default_code', 
                                      readonly=True,
                                      help="Internal reference/SKU from product for CSV matching")