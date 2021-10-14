#date: 2021-10-14T17:07:22Z
#url: https://api.github.com/gists/9cd1b1f5da33936023d7fed6b452784a
#owner: https://api.github.com/users/alexzhu-lyft

# lyftkube ssh -p walletapi -e production
# python

from app.services.wallet import Wallet
wallet = Wallet()

for id in ['6d6c88f28e3071b8eab684735dc161fb',
'0bc3a9523bda31f3a61ae6a84f4aa002',
'7a54bae333fa61abb0dbcf6987d97ce9',
'67e3d6cd206d6a1a75d89fb890eac58c',
'b5e56886f4a6fa1d1cdd9661f9252773',
'f48e241eece983e12ecffd9310e03da6',
'afb00445e6b5619b137cde77c43609a4',
'e760da0a3886fb9372362a03460132cb',
'5a620b08a704914a6402eb384c8ecf6a',
'3843b47a4d02cdc507771dbc64178cce',
'8d79779e6884400083b35ee5ea697d9f',
'c2009365577935813483820b7712ae5b',
'e2fca4bbd76b01b4502d0536bd7cecc4',
'chargeAccount_stripe_card_user_702913265311741954_E33js1otIZCMhqTp,'
'1a1b2017da63fa5cd179a528ca9751a0',
'3b62d028bb7903b704bf4cc70cfa430b']:
  account = wallet.get_charge_account(charge_account_id=id)
  wallet.save_account(account)