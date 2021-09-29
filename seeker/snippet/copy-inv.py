#date: 2021-09-29T17:11:28Z
#url: https://api.github.com/gists/56a86a5495371e3c4a88fa7621ec74e8
#owner: https://api.github.com/users/JulioSerna

with
    parent as (
        select stock_location.id, stock_location.parent_path, sw.id as wh_id, sw.name, stock_location.name as location_parent_name
        from stock_location join stock_warehouse sw
        on stock_location.location_id = sw.view_location_id
        where stock_location.active = True
    ),
  
    location as (
        select parent.parent_path, parent.id as location_parent_id, sl.parent_path,
            sl.id as location_id, parent.wh_id as wh_id,
            sl.name as location_name, parent.name as wh_name,
            parent.location_parent_name
        from stock_location sl join parent
        on (sl.parent_path like parent.parent_path || '%')
        where sl.usage != 'view'
        ),

    product_available as (
        select pp.id as product_id, location.wh_id, location.location_parent_id,
            sum(quantity) as quantity,
            sum(reserved_quantity) as reserved_quantity
        from product_product pp
        join stock_quant sq on sq.product_id = pp.id
        join location on location.location_id = sq.location_id
        group by pp.id, location.wh_id, location.location_parent_id
        ),
    
    product_value as (
        with
            product_avg as (
                select distinct on (product_id) product_id,
                    product_price_history.cost, product_price_history.datetime
                from product_price_history
                order by product_id, product_price_history.datetime desc),
            product_replacement as(
                select distinct pp.id as product_id, ps.price, ps.sequence,
                case
                    when ps.purchase_factor > 0
                        then
                            (ps.price - (ps.price * ps.purchase_factor / 100))
                        else
                            ps.price
                end replacement
                from product_supplierinfo ps
                join product_product pp on pp.id = ps.product_id
                where (ps.date_start <= now()::date AT TIME ZONE 'UTC' and ps.date_end >= now()::date AT TIME ZONE 'UTC')
                or ps.date_end is null
                order by ps.sequence asc),
            rate_currecy as (
                select name, rate as rate
                from res_currency_rate
                where currency_id = 2
                order by name desc
                limit 1
            )
        select pp.id as product_id, uom.name as uom,
            product_avg.cost, product_replacement.replacement,
            product_avg.cost * (select rate from rate_currecy) as usd_cost,
            product_replacement.replacement * (select rate from rate_currecy) as usd_replacement,
            pt.sale_ok, pt.purchase_ok, pt.type, pt.item_edi, pt.gains_manage, pt.status_dmi, lmsc.name as sat_code,
            pp.barcode, pt.name as description, pt.default_seller_id, pp.lifecycle_state, rs.name as sbu,
            pc.name as category, dd.name as division, pb.name as product_brand, pp.default_code
        from product_product pp
        join product_avg on pp.id = product_avg.product_id
        full join product_replacement on pp.id = product_replacement.product_id
        join product_template as pt on pt.id = pp.product_tmpl_id
        join uom_uom as uom on uom.id = pt.uom_id
        left join l10n_mx_edi_product_sat_code lmsc on lmsc.id = pt.l10n_mx_edi_code_sat_id
        left join res_sbu rs on rs.id = pt.sbu_id
        join product_category pc on pc.id = pt.categ_id
        left join division_division dd on dd.id = pc.division_id
        left join product_brand pb on pb.id = pt.product_brand_id
        ),

    product_to_receive as (
        with
            product_receive as (
                select pol.product_id, spt.warehouse_id, spt.default_location_dest_id as location_id,
                    case
                        when pol.product_uom != pt.uom_id then
                            round((pol.qty_pending_receive / (select factor from uom_uom where id=pol.product_uom)) * (select factor from uom_uom where id=pt.uom_id))
                        else
                            pol.qty_pending_receive 
                    end qty_pending_receive
                from purchase_order_line pol
                join purchase_order on purchase_order.id = pol.order_id
                join product_product pp on pp.id = pol.product_id
                join product_template pt on pt.id = pp.product_tmpl_id
                join stock_picking_type spt on spt.id = purchase_order.picking_type_id
                where purchase_order.state in ('purchase', 'done'))
        select product_id, warehouse_id, location_id, sum(qty_pending_receive) as to_receive
        from product_receive
        group by product_id, warehouse_id, location_id),

    product_to_delivery as (
        with
            delivery as(
                select product_id, product_uom, warehouse_id, sw.lot_stock_id as location_id,
                        case
                            when sol.product_uom != pt.uom_id then
                                round((sol.qty_delivered / (select factor from uom_uom where id=sol.product_uom)) * (select factor from uom_uom where id=pt.uom_id))
                            else
                                sol.qty_delivered
                        end qty_uom_delivered,
                        case
                            when sol.product_uom != pt.uom_id then
                                round((sol.product_uom_qty / (select factor from uom_uom where id=sol.product_uom)) * (select factor from uom_uom where id=pt.uom_id))
                            else
                                sol.product_uom_qty
                        end qty_uom
                from sale_order_line sol
                join stock_warehouse sw on sw.id = sol.warehouse_id
                join product_product pp on pp.id = sol.product_id
                join product_template pt on pt.id = pp.product_tmpl_id
                where state in ('sale', 'done'))
            select delivery.product_id, warehouse_id, location_id, sum((qty_uom - qty_uom_delivered)) as to_delivery
            from delivery
            group by product_id, warehouse_id, location_id
            ),

    product_orderpoint as (
        select product_id,  swo.warehouse_id, location_id
        from stock_warehouse_orderpoint swo
        where swo.rotation in ('A', 'AA', 'B')
        and location_id in (select id from parent)
        and active=True
        and swo.id in (
            select id
            from stock_warehouse_orderpoint
            where active=True
            and stock_warehouse_orderpoint.product_id = swo.product_id
            and stock_warehouse_orderpoint.location_id = swo.location_id
            order by create_date desc
            limit 1)
        union
        select pa.product_id, wh_id, location_parent_id
        from product_available pa
        union
        select product_id, warehouse_id, location_id
        from product_to_delivery
        union
        select product_id, warehouse_id, location_id
        from product_to_receive
    ),

    product_warehouse as (
        select po.product_id, po.warehouse_id, po.location_id,
            coalesce(sum(pa.quantity), 0.0) as on_hand,
            coalesce(sum(pa.reserved_quantity), 0.0) as reserved_quantity,
            (select rotation from stock_warehouse_orderpoint swo where swo.location_id=po.location_id and swo.product_id=po.product_id limit 1),
            (select product_min_qty from stock_warehouse_orderpoint swo where swo.location_id=po.location_id and swo.product_id=po.product_id limit 1),
            (select product_max_qty from stock_warehouse_orderpoint swo where swo.location_id=po.location_id and swo.product_id=po.product_id limit 1),
            coalesce(sum(ptd.to_delivery), 0.0) as to_delivery,
            coalesce(sum(ptr.to_receive), 0.0) as to_receive
        from product_orderpoint po
        full join product_available pa on (po.product_id = pa.product_id and po.location_id = pa.location_parent_id)
        full join product_to_delivery ptd on (ptd.product_id = po.product_id and ptd.location_id = po.location_id)
        full join product_to_receive ptr on (ptr.product_id = po.product_id and ptr.location_id = po.location_id)
        group by po.product_id, po.location_id, po.warehouse_id
        order by po.product_id)
select default_code, description, uom, sw.name as wh_name, sl.name as location_name, on_hand, reserved_quantity,
    (cost * on_hand) as cost, (replacement * on_hand) as replacement, (usd_cost * on_hand) as usd_cost, (usd_replacement * on_hand) as usd_replacement,
    rotation, product_min_qty, product_max_qty,
    to_delivery, to_receive,
    pv.sale_ok, pv.purchase_ok, pv.type, pv.item_edi, pv.gains_manage, pv.status_dmi, pv.sat_code,
    pv.barcode, pv.lifecycle_state, pv.sbu,
    pv.category, pv.division, pv.product_brand
from product_value pv
join product_warehouse pw on pw.product_id = pv.product_id
join stock_location sl on sl.id = pw.location_id
join stock_warehouse sw on sw.id = pw.warehouse_id
order by pw.product_id, wh_name



/*
select product_id, uom, cost, replacement, usd_cost, usd_replacement
from product_value
order by product_id
select po.*, ptd.to_delivery
from product_orderpoint po
full join product_to_delivery ptd on (ptd.product_id = po.product_id and ptd.location_id = po.location_id)
full join product_available pa on (po.product_id = pa.product_id and po.location_id = pa.location_parent_id)
where po.product_id = 68
order by warehouse_id
 product_id | warehouse_id | location_id | to_delivery 
------------+--------------+-------------+-------------
         68 |            1 |          12 |   13518.000
         68 |            8 |          85 |       1.000
         68 |           26 |         248 |          11
         
select default_code,  sw.name as wh_name, sl.name as location_name,
    on_hand, reserved_quantity, rotation,
    to_delivery, to_receive
from product_warehouse
join product_product pp on pp.id = product_warehouse.product_id
join stock_location sl on sl.id = product_warehouse.location_id
join stock_warehouse sw on sw.id = product_warehouse.warehouse_id
order by default_code, wh_name
select default_code,  sw.name as wh_name, sl.name as location_name,
    on_hand, reserved_quantity, product_min_qty, product_max_qty, rotation
from product_warehouse
join product_product pp on pp.id = product_warehouse.product_id
join stock_location sl on sl.id = product_warehouse.location_id
join stock_warehouse sw on sw.id = product_warehouse.warehouse_id
order by default_code

select pa.product_id, pa.wh_name, pa.location_id,
    coalesce(sum(pa.quantity), 0.0) as on_hand,
    coalesce(sum(pa.reserved_quantity), 0.0) as reserved_quantity,
    (select rotation from stock_warehouse_orderpoint swo where swo.location_id=pa.location_id and swo.product_id=pa.product_id limit 1),
    (select product_min_qty from stock_warehouse_orderpoint swo where swo.location_id=pa.location_id and swo.product_id=pa.product_id limit 1),
    (select product_max_qty from stock_warehouse_orderpoint swo where swo.location_id=pa.location_id and swo.product_id=pa.product_id limit 1)
from product_orderpoint po
full outer join product_available pa on (po.product_id = pa.product_id and po.location_id = pa.location_parent_id)
group by pa.product_id, wh_name, pa.location_id
order by pa.product_id

product_orderpoint as (
    select product_id,  swo.warehouse_id, location_id, product_min_qty, product_max_qty, rotation
    from stock_warehouse_orderpoint swo
    where swo.rotation in ('A', 'AA', 'B')
    and location_id in (select id from parent)
    and active=True
    and swo.id in (
        select id
        from stock_warehouse_orderpoint
        where active=True
        and stock_warehouse_orderpoint.product_id = swo.product_id
        and stock_warehouse_orderpoint.location_id = swo.location_id
        order by create_date desc
        limit 1)
    union
    select pa.product_id, wh_id, location_parent_id,
        (select product_min_qty from stock_warehouse_orderpoint swo where swo.location_id=location_parent_id and swo.product_id=pa.product_id limit 1),
        (select product_max_qty from stock_warehouse_orderpoint swo where swo.location_id=location_parent_id and swo.product_id=pa.product_id limit 1),
        (select rotation from stock_warehouse_orderpoint swo where swo.location_id=location_parent_id and swo.product_id=pa.product_id limit 1)
    from product_available pa
    order by product_id
)
 product_id | warehouse_id | location_id | product_min_qty | product_max_qty | rotation 
------------+--------------+-------------+-----------------+-----------------+----------
         68 |           18 |         176 |           0.000 |           0.000 | A
         68 |            1 |          12 |           1.000 |         120.000 | AA
         68 |           20 |         194 |         120.000 |         120.000 | A
select po.*
from product_orderpoint po
join product_available pa on (pa.product_id = po.product_id and po.location_id = pa.location_parent_id)

select pa.product_id, wh_name, location_parent_name,
    coalesce(sum(pa.quantity), 0.0) as on_hand,
    coalesce(sum(pa.reserved_quantity), 0.0) as reserved_quantity,
    (select rotation from stock_warehouse_orderpoint swo where swo.location_id=location_parent_id and swo.product_id=pa.product_id limit 1),
    (select product_min_qty from stock_warehouse_orderpoint swo where swo.location_id=location_parent_id and swo.product_id=pa.product_id limit 1),
    (select product_max_qty from stock_warehouse_orderpoint swo where swo.location_id=location_parent_id and swo.product_id=pa.product_id limit 1)
from product_available pa
group by pa.product_id, location_parent_id, wh_name, location_parent_name
order by pa.product_id

select product_warehouse.*,
    coalesce(product_value.cost * on_hand, 0.0) as value_cost,
    coalesce(product_value.replacement * on_hand, 0.0) as value_replacement,
    product_orderpoint.product_min_qty, product_orderpoint.product_max_qty, product_orderpoint.rotation
from product_warehouse
full outer join product_value on product_value.product_id = product_warehouse.product_id
join stock_warehouse sw on sw.id = product_warehouse.wh_id
full outer join product_orderpoint on (
    product_orderpoint.product_id = product_warehouse.product_id and product_orderpoint.location_id = sw.lot_stock_id)
where product_value.product_id = 68
order by product_id, on_hand desc
limit 1;

join product_value on product_value.product_id = product_warehouse.product_id

product_value.cost * product_warehouse.on_hand, product_value.replacement * product_warehouse.on_hand
join product_value on product_value.product_id = product_warehouse.product_id

with
    product_avg as (
        select distinct on (product_id) product_id,  product_price_history.cost, product_price_history.datetime
        from product_price_history
        order by product_id, product_price_history.datetime desc),
    product_replacement as(
        select distinct pp.id as product_id, ps.price, ps.sequence, (ps.price - (ps.price * ps.purchase_factor / 100)) as replacement
        from product_supplierinfo ps
        join product_product pp on pp.id = ps.product_id
        where ps.date_start <= CURRENT_TIMESTAMP and ps.date_end >= CURRENT_TIMESTAMP
        order by ps.sequence asc),
    rate_currecy as (
        select name, rate as rate
        from res_currency_rate
        where currency_id = 2
        order by name desc
        limit 1
    )
select pp.id as product_id, uom.name as uom, product_avg.cost, product_replacement.replacement,
    product_avg.cost * (select rate from rate_currecy) as usd_cost,
    product_replacement.replacement * (select rate from rate_currecy) as usd_replacement,
    pt.sale_ok, pt.purchase_ok, pt.type, pt.item_edi, pt.gains_manage, pt.status_dmi, lmsc.name as sat_code,
    pp.barcode, pt.name as description, pt.default_seller_id, pp.lifecycle_state, rs.name as sbu,
    pc.name as category, dd.name
from product_product pp
join product_avg on pp.id = product_avg.product_id
join product_replacement on pp.id = product_replacement.product_id
join product_template as pt on pt.id = pp.product_tmpl_id
join uom_uom as uom on uom.id = pt.uom_id
left join l10n_mx_edi_product_sat_code lmsc on lmsc.id = pt.l10n_mx_edi_code_sat_id
left join res_sbu rs on rs.id = pt.sbu_id
join product_category pc on pc.id = pt.categ_id
left join division_division dd on dd.id = pc.division_id

select distinct product_price_history.product_id, product_price_history.cost, product_price_history.datetime
from product_price_history
order by product_price_history.datetime desc;


    product_available as (
        select default_code, location.wh_id as wh_id, sq.quantity
        from product_product pp
        join stock_quant sq on sq.product_id = pp.id
        join location on location.location_id = sq.location_id)
select pp.default_code, sw.name, sw.id, sum(pa.quantity) from product_product pp
cross join stock_warehouse sw
full outer join product_available pa on pa.wh_id = sw.id
group by pp.default_code, sw.name, sw.id
order by default_code
*/