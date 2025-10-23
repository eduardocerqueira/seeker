#date: 2025-10-23T17:09:13Z
#url: https://api.github.com/gists/c91850d94ce3186d6918fbd17047b764
#owner: https://api.github.com/users/hugho-ad

# -*- coding: utf-8 -*-

from odoo import models, fields, api


class InheritPurchaseOrderLine(models.Model):
    _inherit = 'purchase.order.line'

    discount_2 = fields.Float(
        string="Descuento 2")
    cost_unit = fields.Float(string="Costo unitario", compute="onchange_disc_y_disc2")

    # adding discount to depends
    @api.depends('product_qty', 'price_unit', 'taxes_id','discount_2')
    def _compute_amount(self):
        for line in self:
            vals = line._prepare_compute_all_values()
            taxes = line.taxes_id.compute_all(
                vals['price_unit'],
                vals['currency_id'],
                vals['product_qty'],
                vals['product'],
                vals['partner'])
            cost_unit = round(line.cost_unit, 2)
            monto = 0
            if line.discount_2:
                desc = (line.price_unit * (1 - line.discount / 100))
                monto = (desc * (1 - line.discount_2 / 100))
            else:
                monto = cost_unit
            line.update({
                'price_tax': sum(t.get('amount', 0.0) for t in taxes.get('taxes', [])),
                'price_total': taxes['total_included'],
                'price_subtotal': monto * line.product_qty,
            })

    def _prepare_compute_all_values(self):
        # Hook method to returns the different argument values for the
        # compute_all method, due to the fact that discounts mechanism
        # is not implemented yet on the purchase orders.
        # This method should disappear as soon as this feature is
        # also introduced like in the sales module.
        self.ensure_one()
        return {
            'price_unit': self.cost_unit,
            'currency_id': self.order_id.currency_id,
            'product_qty': self.product_qty,
            'product': self.product_id,
            'partner': self.order_id.partner_id,
        }

    @api.onchange('discount','discount_2','price_unit','product_qty')
    def onchange_disc_y_disc2(self):
        for record in self:
            desc = 0
            if record.discount:
                desc = (record.price_unit * (1 - record.discount / 100))
            if record.discount_2:
                if desc == 0:
                    desc = record.price_unit
                desc = (desc * (1 - record.discount_2 / 100))
            record.cost_unit = desc or record.price_unit

    def _prepare_account_move_line(self):
        res = super(InheritPurchaseOrderLine, self)._prepare_account_move_line()
        res.update({
            'discount_2': self.discount_2
        })
        return res
