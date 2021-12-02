#date: 2021-12-02T16:56:29Z
#url: https://api.github.com/gists/3c39b57c0ddb8c265d5c2019fc673185
#owner: https://api.github.com/users/mast22

from decimal import Decimal
from django.db import models as m
from django.utils.translation import gettext_lazy as __
from rest_framework_json_api import serializers as s
from django.core.exceptions import ObjectDoesNotExist

from apps.common.models import Model
from apps.partners.models import Outlet
from apps.banks.models import CreditProduct, ExtraService
from .. import const as c
from ..const import CreditProductStatus
from ..workflows import OrderStatusFSM


class Order(OrderStatusFSM, Model):
    status = m.CharField(
        verbose_name=__('Статус'),
        choices=c.OrderStatus.as_choices(),
        default=c.OrderStatus.NEW,
        max_length=c.OrderStatus.length()
    )