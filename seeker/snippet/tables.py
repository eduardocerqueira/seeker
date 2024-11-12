#date: 2024-11-12T17:01:07Z
#url: https://api.github.com/gists/c768bd61cef534c623cdc6dd1b5cc862
#owner: https://api.github.com/users/serg-yalosovetsky

from enum import Enum
from piccolo.table import Table
from piccolo.columns import (
    Varchar,
    Integer,
    BigInt,
    Boolean,
    Timestamp,
    ForeignKey,
    Text,
    Numeric,
    Array,
    Float,
    JSON,
    Serial,
    Timestamptz,
)
from piccolo.columns.base import OnDelete, OnUpdate
from piccolo.columns.defaults import TimestampNow
from datetime import datetime, timedelta
from decimal import Decimal
from piccolo.columns.defaults.timestamptz import TimestamptzNow



def get_expiry_time(hours: int = 24, minutes: int = 0, seconds: int = 0):
    return datetime.now() + timedelta(
        hours=hours, minutes=minutes, seconds=seconds
    )


class DisputeStatus(Table, tablename="dispute_status"):
    """
    Таблица типов статусов споров
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)


class AmlStatus(Table, tablename="aml_status"):
    """
    Таблица типов статусов AML
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)

class KycStatus(Table, tablename="kyc_status"):
    """
    Таблица типов статусов KYC
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)

class PaymentStatus(Table, tablename="payment_status"):
    """
    Таблица типов статусов платежей
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)

class OrderStatus(Table, tablename="order_status"):
    """
    Таблица типов статусов заказов
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)

class VerificationRequirement(Table, tablename="verification_requirement"):
    """
    Таблица типов требований к покупателю
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)

class AdvertisementType(Table, tablename="advertisement_type"):
    """
    Таблица типов объявлений, buy or sell
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)


class TransactionStatus(Table, tablename="transaction_status"):
    """
    Таблица типов статусов транзакций
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)

class AdvertisementStatus(Table, tablename="advertisement_status"):
    """
    Таблица типов статусов заказов
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False, description="Active, Paused, Deleted, Closed")

class TransactionType(Table, tablename="transaction_type"):
    """
    Таблица типов транзакций
    """
    id = Integer(primary_key=True)
    title = Varchar(unique=True, null=False)

class FiatCurrencies(Table, tablename="fiat_currencies"):
    """
    Table for storing fiat currency information.
    """
    id = Integer(primary_key=True)
    title = Varchar(length=10, null=False)
    symbol = Varchar(length=10, null=False)
    iso = Varchar(length=3, null=False)


class CryptoCurrencies(Table, tablename="crypto_currencies"):
    """
    Table for storing cryptocurrency information.
    """
    id = Integer(primary_key=True)
    title = Varchar(length=10, null=False)
    symbol = Varchar(length=10, null=False)
    iso = Varchar(length=3, null=False)

class PaymentMethods(Table, tablename="payment_methods"):
    """
    Table for storing payment methods information. (like mono, privat and payoneer)
    """
    id = Integer(primary_key=True)
    title = Varchar(length=10, null=False)
    description = Text(null=True)


class Countries(Table, tablename="countries"):
    """
    Table for storing country information.
    """
    id = Integer(primary_key=True)
    title = Varchar(length=100, null=False)
    capital = Varchar(length=100, null=False)
    iso2 = Varchar(length=2, null=False)
    iso3 = Varchar(length=3, null=False)


class Users(Table, tablename="users"):
    """Таблица пользователей"""
    id = Integer(primary_key=True, index=True)
    first_name = Varchar(length=100, null=False, index=True)
    telegram_id = BigInt(null=False, index=True)
    username = Varchar(length=100, null=True)
    photo_url = Text(null=True)
    auth_date = Timestamp(default=TimestampNow(), null=False, timezone=True)
    created_at = Timestamp(default=TimestampNow(), timezone=True)
    locked = Boolean(default=False, null=False)
    is_active = Boolean(default=True, null=True)

    # Security fields
    secret_code = "**********"=32, null=True)
    mnemonic = Array(base_column=Varchar(), null=True)
    email = Varchar(length=50, null=True)
    code = Integer(null=True)
    expiration_time = Timestamp(null=True, timezone=True)
    
    # Authentication fields
    pin_code = Array(base_column=Integer(), null=True)
    email_verif = Boolean(default=False, null=False)
    secret_key = "**********"=32, null=True)
    sec_key_verif = Boolean(default=False, null=False)
    count_entering_pincode = Integer(default=0, null=False)
    count_entering_2fa = Integer(default=0, null=False)
    
    # Status fields
    reputation_score = Float(default=0.0)
    kyc_status = ForeignKey(references=KycStatus)
    kyc_verified_at = Timestamp(null=True, timezone=True)
    aml_status = ForeignKey(references=AmlStatus)
    aml_verified_at = Timestamp(null=True, timezone=True)
    is_mediator = Boolean(default=False, null=False)

    is_admin = Boolean(default=False, null=False)


class Wallets(Table, tablename="wallets"):
    """Таблица кошельков"""
    id = Integer(primary_key=True)
    user_id = ForeignKey(references=Users, null=False)
    address = Varchar(length=34, null=False, unique=True)
    locked = Numeric(digits=(20, 6), null=False, default=Decimal(0))
    currency = ForeignKey(references=CryptoCurrencies, null=False)
    is_active = Boolean(default=True)

class Transactions(Table, tablename="transactions"):
    """
    Таблица транзакций
    """
    id = Integer(primary_key=True)
    amount = Integer(null=False)
    currency = Varchar(length=3, null=False)
    timestamp = Timestamp(default=TimestampNow())
    description = Varchar(null=True)
    sender_address = Varchar(length=34, null=False)
    receiver_address = Varchar(length=34, null=False)
    type = ForeignKey(references=TransactionType, null=False)
    wallet_id = ForeignKey(references=Wallets, 
                           null=False, index=True, on_delete=OnDelete.cascade)


class TransactionStatusJournal(Table, tablename="transaction_status_journal"):
    """
    Таблица журнала статусов транзакций
    """
    id = Integer(primary_key=True)
    transaction_id = ForeignKey(references=Transactions, null=False)
    status = ForeignKey(references=TransactionStatus, null=False)
    status_set_at = Timestamp(default=TimestampNow())


class Advertisement(Table, tablename="advertisements"):
    id = Serial(primary_key=True)
    ads_type = ForeignKey(references=AdvertisementType, description="Advertisement type - sell or buy")
    seller = ForeignKey(references=Users, description="Seller ID")
    wallet = ForeignKey(references=Wallets, description="Seller's wallet ID") 
    fiat_currency = ForeignKey(references=FiatCurrencies, description="Fiat currency ID")
    crypto_currency = ForeignKey(references=CryptoCurrencies, description="Cryptocurrency ID")
    buyer_requirements = ForeignKey(references=VerificationRequirement, description="Buyer requirements")
    minimum_rating = Float(default=0.0)
    total_amount = Numeric(decimal_places=8)
    min_amount = Numeric(decimal_places=2)
    max_amount = Numeric(decimal_places=2)
    is_active = Boolean(default=True)
    price_per_unit = Numeric(decimal_places=2)
    time_limit_seconds = Integer(default=30 * 60)
    terms_of_use = Text(null=True)
    remarks = Text(null=True)
    country = ForeignKey(references=Countries, description="ID country")
    status = ForeignKey(references=AdvertisementStatus, null=False)
    payment_methods = Array(base_column=Integer(), default=[])
    created_at = Timestamptz(default=TimestamptzNow())
    expires_at = Timestamptz(default=get_expiry_time())
    updated_at = Timestamptz(default=TimestamptzNow())


class Order(Table, tablename="orders"):
    id = Serial(primary_key=True)
    advertisement = ForeignKey(references=Advertisement)
    buyer = ForeignKey(references=Users)
    amount = Numeric(decimal_places=8)
    order_status = ForeignKey(references=OrderStatus)
    wallet = ForeignKey(references=Wallets)
    fiat_price = Numeric(decimal_places=2)
    crypto_price = Numeric(decimal_places=8)
    created_at = Timestamptz(default=TimestamptzNow())
    expires_at = Timestamptz()
    completed_at = Timestamptz(null=True)


class Payments(Table, tablename="payments"):
    id = Serial(primary_key=True)
    description = Varchar(length=500, null=True)
    verified = Boolean(default=False)
    advertisement = ForeignKey(references=Advertisement)
    payment_method = ForeignKey(references=PaymentMethods)
    order = ForeignKey(references=Order)
    data = JSON(null=True)
    payment_status = ForeignKey(references=PaymentStatus)
    created_at = Timestamptz(default=TimestamptzNow())
    completed_at = Timestamptz(null=True)

# Order.payments = ManyToMany(references=Payments, back_reference='order')

class Dispute(Table, tablename="disputes"):
    id = Serial(primary_key=True)
    order = ForeignKey(references=Order)
    initiated_by = ForeignKey(references=Users)
    reason = Varchar(length=500)
    evidence = JSON()  # Store paths to uploaded evidence files
    resolution = Varchar(length=500, null=True)
    mediator = ForeignKey(references=Users, null=True)
    status = ForeignKey(references=DisputeStatus)
    created_at = Timestamptz(default=TimestamptzNow())
    resolved_at = Timestamptz(null=True)


class DisputeMessage(Table, tablename="dispute_messages"):
    id = Serial(primary_key=True)
    dispute = ForeignKey(references=Dispute)
    sender = ForeignKey(references=Users)
    message = Varchar(length=1000)
    attachments = JSON(null=True)  # Store paths to uploaded files
    created_at = Timestamptz(default=TimestamptzNow())

    
class Logs(Table, tablename="logs"):
    id = Serial(primary_key=True)
    order = ForeignKey(references=Order)
    action = Varchar(length=100)
    actor = ForeignKey(references=Users)
    details = JSON(null=True)
    created_at = Timestamptz(default=TimestamptzNow())
())
