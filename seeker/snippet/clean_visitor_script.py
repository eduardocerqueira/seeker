#date: 2021-10-01T01:28:02Z
#url: https://api.github.com/gists/12212c18c5bbd356cdbed94032bc4b13
#owner: https://api.github.com/users/bnmounir

from spots.models import Visitor, ParkingSpot
from django.db.models import Count

visitors = Visitor.objects.all()

for visitor in visitors:
    if not visitor.plate or not visitor.spot:
        visitor.delete()


duplicated_visitor_data = Visitor.objects.values('plate', 'spot').annotate(Count('id')).order_by().filter(id__count__gt=1)

for data in duplicated_visitor_data:
    spot = ParkingSpot.objects.get(pk=data["spot"])
    plate = data["plate"]
    visitors_to_update = Visitor.objects.filter(spot=spot, plate=plate).order_by('created_at')
    clean_visitor = Visitor.objects.create(spot=spot, plate=plate)
    for visitor in visitors_to_update:
        if visitor.phone:
            clean_visitor.phone = visitor.phone
        if visitor.codes:
            clean_visitor.codes.set(visitor.codes.all())
    v = visitors_to_update[0]
    v.codes.set(clean_visitor.codes.all())
    v.phone = clean_visitor.phone
    visitors_to_delete = Visitor.objects.filter(pk__in=Visitor.objects.filter(spot=spot, plate=plate)[1:]).order_by('created_at').delete()
    v.save()