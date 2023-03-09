#date: 2023-03-09T16:47:34Z
#url: https://api.github.com/gists/2175dbe8682753dfb574c19b8b9c9ae6
#owner: https://api.github.com/users/ninjaguru-git

import math

from rest_framework import serializers
from django.core.cache import cache

from notifications.api.serializers import NotificationSentSerializer
from notifications.models import Notification, NotificationSent
from orders.api.fields import Base64ImageField
from orders.models import (
    Order,
    ApprovalDocuments,
    ApprovalImage,
    Product,
    GeneratedDocuments,
    GeneratedFile,
    Note,
    Coupon,
    ProductOrder,
    OrderViewer,
    OrderStatusUpdateLog,
    CustomReceipt,
    Item,
    ItemFeature,
    ItemImage,
)
from payment.models import CheckoutState
from orders.settings import (
    TYPE_STATUS_WAITING_ON_CUSTOMER,
    TYPE_PROCESSING_TIME_0_30,
    TYPE_PROCESSING_TIME_2,
    CURRENCY_CHOICES,
)
from payment.api.serializers import MerchantSerializers
from shipping.api.serializers import ShippingSerializer
from users.api.serializers import CustomerSerializerLite, UserSerializerLite
from ida.celery import app


class ApprovalImageFullSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApprovalImage
        fields = "__all__"


class ApprovalDocumentsFullSerializer(serializers.ModelSerializer):
    profile = ApprovalImageFullSerializer()
    license_front = ApprovalImageFullSerializer()
    license_back = ApprovalImageFullSerializer()
    signature = ApprovalImageFullSerializer()

    class Meta:
        model = ApprovalDocuments
        fields = "__all__"


class GeneratedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneratedFile
        exclude = ["modified"]


class GeneratedDocumentsSerializer(serializers.ModelSerializer):
    id_card_front = GeneratedFileSerializer()
    id_card_back = GeneratedFileSerializer()
    booklet = GeneratedFileSerializer()
    packing_slip = GeneratedFileSerializer()
    a4_document = GeneratedFileSerializer()
    receipt = GeneratedFileSerializer()
    customer_data = GeneratedFileSerializer()
    shipping_label = GeneratedFileSerializer()

    class Meta:
        model = GeneratedDocuments
        exclude = ["modified"]


class ProductCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = [
            "id",
            "price",
            "name",
            "category",
            "years",
            "shipping",
            "show_at_checkout",
            "policy",
            "free_count",
            "img",
        ]


class ProductSerializer(serializers.ModelSerializer):
    price = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = [
            "id",
            "price",
            "name",
            "category",
            "years",
            "shipping",
            "show_at_checkout",
            "policy",
            "free_count",
            "img",
        ]

    def get_price(self, product: Product):
        request = self.context.get("request")
        if request and request.query_params.get("currency"):
            allowed_currencies = {i[0] for i in CURRENCY_CHOICES}
            currency = request.query_params.get("currency").upper()
            if currency in allowed_currencies:
                rate = cache.get(f"{currency}_USD_RATE")
                return int(math.ceil(float(product.price.amount) / rate))

        return product.price.amount


class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = Item
        fields = [
            "id",
            "product",
            "description",
        ]


class ItemFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = ItemFeature
        fields = [
            "id",
            "item",
            "feature",
        ]


class ItemImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ItemImage
        fields = [
            "id",
            "item",
            "image",
        ]


class NoteSerializer(serializers.ModelSerializer):
    owner = UserSerializerLite(source="user", read_only=True)

    class Meta:
        model = Note
        fields = ["id", "order", "user", "text", "created", "modified", "owner"]


class ProductSerializerLite(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ["id", "name", "price", "category", "show_at_checkout"]


class ProductOrderSerializer(serializers.ModelSerializer):
    price = serializers.SerializerMethodField()
    original_price = serializers.SerializerMethodField()

    class Meta:
        model = ProductOrder
        fields = ["id", "price", "product", "original_price"]

    def create(self, validated_data):
        product_order = super().create(validated_data)
        request = self.context.get("request")
        if request:
            order = Order.objects.get(id=request.query_params["order_id"])
            if not order.products.filter(id=product_order.id).exists():
                order.products.add(product_order)
        return product_order

    def get_price(self, product: ProductOrder):
        request = self.context.get("request")
        if request and request.query_params.get("currency"):
            allowed_currencies = {i[0] for i in CURRENCY_CHOICES}
            currency = request.query_params.get("currency").upper()
            if currency in allowed_currencies:
                rate = cache.get(f"{currency}_USD_RATE")
                return int(math.ceil(float(product.price.amount) / rate))

        return product.price.amount

    def get_original_price(self, product: ProductOrder):
        request = self.context.get("request")
        if request and request.query_params.get("currency"):
            allowed_currencies = {i[0] for i in CURRENCY_CHOICES}
            currency = request.query_params.get("currency").upper()
            if currency in allowed_currencies:
                rate = cache.get(f"{currency}_USD_RATE")
                return int(math.ceil(float(product.original_price.amount) / rate))

        return product.original_price.amount


class OrderListSerializer(serializers.ModelSerializer):
    product_type = serializers.ReadOnlyField()
    photo = serializers.ReadOnlyField(source="photo_status")
    license_status = serializers.ReadOnlyField(source="license_approved_status")
    signature = serializers.ReadOnlyField(source="signature_status")
    is_urgent = serializers.ReadOnlyField()
    idp_shipped = serializers.ReadOnlyField()
    sent_to_print = serializers.ReadOnlyField()
    idp_emailed = serializers.ReadOnlyField()
    paid_date = serializers.ReadOnlyField()
    courier = serializers.SerializerMethodField()

    class Meta:
        model = Order
        fields = [
            "id",
            "status",
            "customer",
            "created",
            "completed_date",
            "processing_date",
            "photo",
            "license_status",
            "signature",
            "processing_time",
            "processing_time_status",
            "processing_used_time",
            "processing_time_last_start_date",
            "idp_emailed",
            "sent_to_print",
            "idp_shipped",
            "is_urgent",
            "product_type",
            "total",
            "woo_created",
            "first_name",
            "last_name",
            "paid_date",
            "currency",
            "courier",
        ]

    def get_courier(self, order: Order):
        if order.shipping:
            return order.shipping.shipping_method
        else:
            return None


class CouponSerializer(serializers.ModelSerializer):
    products = serializers.PrimaryKeyRelatedField(
        queryset=Product.objects.all(), required=False, allow_null=True, many=True
    )

    class Meta:
        model = Coupon
        exclude = ["modified"]


class CouponClientSerializer(serializers.ModelSerializer):
    products = serializers.PrimaryKeyRelatedField(
        queryset=Product.objects.all(), required=False, allow_null=True, many=True
    )
    amount = serializers.SerializerMethodField()

    class Meta:
        model = Coupon
        exclude = ["modified"]

    def get_amount(self, coupon: Coupon):
        request = self.context.get("request")
        if coupon.discount_type == Coupon.DISCOUNT_PERCENTAGE:
            return coupon.amount

        if request and request.query_params.get("currency"):
            allowed_currencies = {i[0] for i in CURRENCY_CHOICES}
            currency = request.query_params.get("currency").upper()
            if currency in allowed_currencies:
                rate = cache.get(f"{currency}_USD_RATE")
                return int(math.ceil(float(coupon.amount) / rate))

        return coupon.amount


class CouponSerializerLite(serializers.ModelSerializer):
    amount = serializers.SerializerMethodField()

    class Meta:
        model = Coupon
        fields = ["id", "free_shipping", "discount_type", "amount", "name", "description", "products"]

    def get_amount(self, coupon: Coupon):
        request = self.context.get("request")
        if coupon.discount_type == Coupon.DISCOUNT_PERCENTAGE:
            return coupon.amount

        if request and request.query_params.get("currency"):
            allowed_currencies = {i[0] for i in CURRENCY_CHOICES}
            currency = request.query_params.get("currency").upper()
            if currency in allowed_currencies:
                rate = cache.get(f"{currency}_USD_RATE")
                return int(math.ceil(float(coupon.amount) / rate))

        return coupon.amount


class OrderStatusUpdateLogsSerializer(serializers.ModelSerializer):
    user = UserSerializerLite(read_only=True)

    class Meta:
        model = OrderStatusUpdateLog
        fields = ["id", "user", "status", "created_at"]


class OrderViewerSerializer(serializers.ModelSerializer):
    user = UserSerializerLite(read_only=True)

    class Meta:
        model = OrderViewer
        fields = ("user", "viewed_at")


class OrderSerializer(serializers.ModelSerializer):
    product_type = serializers.ReadOnlyField()
    is_urgent = serializers.ReadOnlyField()
    idp_shipped = serializers.ReadOnlyField()
    sent_to_print = serializers.ReadOnlyField()
    idp_emailed = serializers.ReadOnlyField()
    paid_date = serializers.ReadOnlyField()
    approval_documents = ApprovalDocumentsFullSerializer()
    generated_documents = GeneratedDocumentsSerializer()
    coupon = CouponSerializer()
    customer = CustomerSerializerLite()
    shipping = ShippingSerializer()
    merchant = MerchantSerializers()
    checkout_token = "**********"
    notifications = serializers.SerializerMethodField()
    viewers = serializers.SerializerMethodField()
    status_updates = OrderStatusUpdateLogsSerializer(source="order_status_update_logs", many=True)

    class Meta:
        model = Order
        exclude = ["dump_data", "fbtrace_id", "manychat_id"]

    def update(self, instance: Order, validated_data: dict):
        request = self.context.get("request")
        if request:
            user = request.user
            if (
                user.is_authenticated
                and user.is_admin
                and validated_data.get("status")
                and validated_data["status"] != instance.status
            ):
                OrderStatusUpdateLog.objects.create(order=instance, user=user, status=validated_data["status"])

        return super().update(instance, validated_data)

    def get_notifications(self, obj):
        notifications = {}
        for notification in Notification.objects.all():
            notification_sent = (
                NotificationSent.objects.filter(order_id=obj.id, notification=notification).order_by("-created").first()
            )
            notifications[notification.unique_id] = (
                NotificationSentSerializer(notification_sent).data if notification_sent else None
            )
        return notifications

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"c "**********"h "**********"e "**********"c "**********"k "**********"o "**********"u "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"o "**********"b "**********"j "**********") "**********": "**********"
        if hasattr(obj, "order_checkouts") and obj.order_checkouts:
            return obj.order_checkouts.token

    def get_viewers(self, obj):
        request = self.context.get("request", None)
        viewer_records = OrderViewer.objects.filter(order=obj)

        if request and request.user.is_authenticated:
            viewer_records = viewer_records.exclude(user=request.user)

        return OrderViewerSerializer(viewer_records, many=True).data


class ApprovalDocumentsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApprovalDocuments
        fields = ["id", "profile", "license_front", "license_back", "signature"]


class ApprovalImageSerializer(serializers.ModelSerializer):
    cropped_image = Base64ImageField(max_length=None, use_url=True, required=False, allow_null=True, default=None)
    image = Base64ImageField(max_length=None, use_url=True)
    order = serializers.PrimaryKeyRelatedField(
        queryset=Order.objects.all(), write_only=True, required=False, allow_null=True
    )
    checkout = serializers.PrimaryKeyRelatedField(
        queryset=CheckoutState.objects.all(), write_only=True, required=False, allow_null=True
    )

    class Meta:
        model = ApprovalImage
        fields = ["id", "category", "status", "image", "cropped_image", "crop_box_data", "order", "checkout"]

    def create_update_approval_documents(self, instance, order, checkout):
        # linked_instance can be Order instance or CheckoutState instance
        linked_instance = order or checkout
        approval_documents = linked_instance.approval_documents
        if not approval_documents:
            approval_documents = ApprovalDocuments.objects.create()
            linked_instance.approval_documents = approval_documents
            linked_instance.save()
        setattr(approval_documents, instance.category, instance)
        approval_documents.save()

    def update(self, instance, validated_data):
        try:
            order = validated_data.pop("order")
        except KeyError:
            order = None
        try:
            checkout = validated_data.pop("checkout")
        except KeyError:
            checkout = None
        instance = super().update(instance, validated_data)
        self.create_update_approval_documents(instance, order, checkout)
        return instance

    def create(self, validated_data):
        try:
            order = validated_data.pop("order")
        except KeyError:
            order = None
        try:
            checkout = validated_data.pop("checkout")
        except KeyError:
            checkout = None
        instance = super().create(validated_data)
        self.create_update_approval_documents(instance, order, checkout)
        return instance

    


class OrderCheckoutSerializer(serializers.ModelSerializer):
    status = serializers.CharField(write_only=True, default=TYPE_STATUS_WAITING_ON_CUSTOMER)
    processing_time = serializers.DurationField(
        write_only=True, source="get_processing_time", required=False, allow_null=True
    )

    class Meta:
        model = Order
        exclude = ["dump_data"]

    def get_processing_time(self, obj):
        return TYPE_PROCESSING_TIME_0_30 if obj.is_urgent else TYPE_PROCESSING_TIME_2


class CustomReceiptSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomReceipt
        fields = "__all__"
        read_only_fields = ["receipt"]

    def create(self, validated_data):
        instance = super().create(validated_data)
        app.send_task("orders.tasks.generate_receipt_custom_task", args=(instance.id,))
        app.send_task("orders.tasks.update_or_create_xero_invoice", args=(instance.order.id,))
        return instance


class PrepopulatedCheckoutSerializer(serializers.Serializer):
    order = serializers.PrimaryKeyRelatedField(queryset=Order.objects.all())
    email = serializers.EmailField()