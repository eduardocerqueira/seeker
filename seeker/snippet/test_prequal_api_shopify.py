#date: 2025-04-03T16:59:23Z
#url: https://api.github.com/gists/9879d969f74b47944c765d0fe1001073
#owner: https://api.github.com/users/oleksii-fedorenko-94

import re

import pytest
from aiohttp import BasicAuth
from chameleon_toolkit import Profile
from thor.pba.resources.utils.payload_utils import generate_create_merchant_payload

from affirm.test_framework.api.endpoints.api.pba.v1.pba_api import PbaAPI


BAPI_US_CHAMELEON_PROFILE = Profile.minimal_profile(name="BAPI US Chameleon Profile")
BUYER_ID_REGEX_PATTERN = r"(\w{4}-\w{4}-\w{4})"


@pytest.mark.asyncio
@pytest.mark.us
@pytest.mark.non_blocking
@pytest.mark.ci_ready
@pytest.mark.parametrize("chameleon_profile", ([BAPI_US_CHAMELEON_PROFILE]))
async def test_201_create_prequal(
    fxt_create_shopify_partner_and_fp,
    fxt_create_partner_and_fp,
    fxt_generate_create_buyer_payload,
    fxt_chameleon_profile,
    fxt_pba_api: PbaAPI,
    fxt_pba_merchants_api,
    fxt_generate_create_prequal_payload,
    fxt_poll_until_prequal_is_not_pending,
    fxt_create_rule_for_shopify_merchant,
):
    # Get auth and uuid credentials
    partner_public_api_key = fxt_create_shopify_partner_and_fp.partner.public_key
    partner_private_api_key = fxt_create_shopify_partner_and_fp.partner.private_key
    financing_package_uuid = fxt_create_shopify_partner_and_fp.financing_package_uuid
    financing_program_uuid = fxt_create_shopify_partner_and_fp.financing_program_uuid

    partner_auth = BasicAuth(partner_public_api_key, partner_private_api_key)

    # Create merchant
    create_merchant_payload = generate_create_merchant_payload(
        fxt_chameleon_profile,
        business_name="Shopify",
        financing_package_uuid=financing_package_uuid,
    )
    create_new_merchant_response = await fxt_pba_merchants_api.create_merchant(
        payload=create_merchant_payload, auth=partner_auth
    )
    merchant_id = create_new_merchant_response.payload.id

    assert create_new_merchant_response.payload.status is not None
    await create_new_merchant_response.assert_ok()

    # Update merchant FP rule to make it Adaptive and prequal enabled
    update_merchant_rule = await fxt_create_rule_for_shopify_merchant(merchant_id, financing_program_uuid)
    assert update_merchant_rule is not None

    # Create buyer
    create_buyer_payload = fxt_generate_create_buyer_payload(
        chameleon_profile=fxt_chameleon_profile, annual_income=1234567
    )
    create_buyer_response = await fxt_pba_api.create_buyer(payload=create_buyer_payload, auth=partner_auth)
    buyer_id = create_buyer_response.payload.id

    assert re.match(BUYER_ID_REGEX_PATTERN, buyer_id), "ID does not match the regex pattern"
    assert create_buyer_response.payload.status == "new"

    # Create prequal
    create_prequal_payload = fxt_generate_create_prequal_payload(buyer_id=buyer_id, merchant_id=merchant_id)
    create_prequal_response = await fxt_pba_api.create_prequal(payload=create_prequal_payload, auth=partner_auth)
    created_prequal_id = create_prequal_response.payload.id
    assert created_prequal_id is not None


@pytest.mark.asyncio
@pytest.mark.us
@pytest.mark.non_blocking
@pytest.mark.ci_ready
@pytest.mark.parametrize("chameleon_profile", ([BAPI_US_CHAMELEON_PROFILE]))
async def test_200_prequal_approved(
    fxt_create_shopify_partner_and_fp,
    fxt_create_partner_and_fp,
    fxt_generate_create_buyer_payload,
    fxt_chameleon_profile,
    fxt_pba_api: PbaAPI,
    fxt_pba_merchants_api,
    fxt_generate_create_prequal_payload,
    fxt_poll_until_prequal_is_not_pending,
    fxt_create_rule_for_shopify_merchant,
):
    # Get auth and uuid credentials
    partner_public_api_key = fxt_create_shopify_partner_and_fp.partner.public_key
    partner_private_api_key = fxt_create_shopify_partner_and_fp.partner.private_key
    financing_package_uuid = fxt_create_shopify_partner_and_fp.financing_package_uuid
    financing_program_uuid = fxt_create_shopify_partner_and_fp.financing_program_uuid

    partner_auth = BasicAuth(partner_public_api_key, partner_private_api_key)

    # Create merchant
    create_merchant_payload = generate_create_merchant_payload(
        fxt_chameleon_profile,
        business_name="Shopify",
        financing_package_uuid=financing_package_uuid,
    )
    create_new_merchant_response = await fxt_pba_merchants_api.create_merchant(
        payload=create_merchant_payload, auth=partner_auth
    )
    merchant_id = create_new_merchant_response.payload.id

    assert create_new_merchant_response.payload.status is not None
    await create_new_merchant_response.assert_ok()

    # Update merchant FP rule to make it Adaptive and prequal enabled
    update_merchant_rule = await fxt_create_rule_for_shopify_merchant(merchant_id, financing_program_uuid)
    assert update_merchant_rule is not None

    # Create buyer
    create_buyer_payload = fxt_generate_create_buyer_payload(
        chameleon_profile=fxt_chameleon_profile, annual_income=1234567
    )
    create_buyer_response = await fxt_pba_api.create_buyer(payload=create_buyer_payload, auth=partner_auth)
    buyer_id = create_buyer_response.payload.id

    assert re.match(BUYER_ID_REGEX_PATTERN, buyer_id), "ID does not match the regex pattern"
    assert create_buyer_response.payload.status == "new"

    # Create prequal
    create_prequal_payload = fxt_generate_create_prequal_payload(buyer_id=buyer_id, merchant_id=merchant_id)
    create_prequal_response = await fxt_pba_api.create_prequal(payload=create_prequal_payload, auth=partner_auth)
    created_prequal_id = create_prequal_response.payload.id

    # TODO: Clarify why we are getting HTTP 500 Internal Server Error instead of HTTP 200 OK with "prequal_approved"
    # Get prequal
    get_prequal_response = await fxt_poll_until_prequal_is_not_pending(created_prequal_id, auth=partner_auth)

    assert get_prequal_response.payload.prequal_decision == "prequal_approved"
    assert get_prequal_response.payload.universal_prequal_decision == "prequal_approved"
    assert get_prequal_response.payload.universal_prequal is not None
    assert get_prequal_response.payload.prequalification is not None
    assert get_prequal_response.payload.buyer_id == buyer_id
    assert get_prequal_response.payload.prequal_id == created_prequal_id
    assert get_prequal_response.payload.merchant_id == merchant_id


@pytest.mark.asyncio
@pytest.mark.us
@pytest.mark.non_blocking
@pytest.mark.ci_ready
@pytest.mark.parametrize("chameleon_profile", ([BAPI_US_CHAMELEON_PROFILE]))
async def test_200_prequal_declined(
    fxt_create_shopify_partner_and_fp,
    fxt_create_partner_and_fp,
    fxt_generate_create_buyer_payload,
    fxt_chameleon_profile,
    fxt_pba_api: PbaAPI,
    fxt_pba_merchants_api,
    fxt_generate_create_prequal_payload,
    fxt_poll_until_prequal_is_not_pending,
    fxt_create_rule_for_shopify_merchant,
):
    # Get auth and uuid credentials
    partner_public_api_key = fxt_create_shopify_partner_and_fp.partner.public_key
    partner_private_api_key = fxt_create_shopify_partner_and_fp.partner.private_key
    financing_package_uuid = fxt_create_shopify_partner_and_fp.financing_package_uuid
    financing_program_uuid = fxt_create_shopify_partner_and_fp.financing_program_uuid

    partner_auth = BasicAuth(partner_public_api_key, partner_private_api_key)

    # Create merchant
    create_merchant_payload = generate_create_merchant_payload(
        fxt_chameleon_profile,
        business_name="Shopify",
        financing_package_uuid=financing_package_uuid,
    )
    create_new_merchant_response = await fxt_pba_merchants_api.create_merchant(
        payload=create_merchant_payload, auth=partner_auth
    )
    merchant_id = create_new_merchant_response.payload.id

    assert create_new_merchant_response.payload.status is not None
    await create_new_merchant_response.assert_ok()

    # Update merchant FP rule to make it Adaptive and prequal enabled
    update_merchant_rule = await fxt_create_rule_for_shopify_merchant(merchant_id, financing_program_uuid)
    assert update_merchant_rule is not None

    # Create buyer
    create_buyer_payload = fxt_generate_create_buyer_payload(chameleon_profile=fxt_chameleon_profile, annual_income=10)
    create_buyer_response = await fxt_pba_api.create_buyer(payload=create_buyer_payload, auth=partner_auth)
    buyer_id = create_buyer_response.payload.id

    assert re.match(BUYER_ID_REGEX_PATTERN, buyer_id), "ID does not match the regex pattern"
    assert create_buyer_response.payload.status == "new"

    # Create prequal
    create_prequal_payload = fxt_generate_create_prequal_payload(
        buyer_id=buyer_id, merchant_id=merchant_id, total_amount=100, unit_price=1000
    )

    create_prequal_response = await fxt_pba_api.create_prequal(payload=create_prequal_payload, auth=partner_auth)

    created_prequal_id = create_prequal_response.payload.id

    # TODO: Clarify why we are getting HTTP 500 Internal Server Error instead of HTTP 200 OK with "prequal_declined"
    # Get prequal
    get_prequal_response = await fxt_poll_until_prequal_is_not_pending(created_prequal_id, auth=partner_auth)

    assert get_prequal_response.payload.prequal_decision == "prequal_declined"
    assert get_prequal_response.payload.universal_prequal_decision == "prequal_declined"
    assert get_prequal_response.payload.universal_prequal is None
    assert get_prequal_response.payload.prequalification is None
    assert get_prequal_response.payload.buyer_id == buyer_id
    assert get_prequal_response.payload.prequal_id == created_prequal_id
    assert get_prequal_response.payload.merchant_id == merchant_id
