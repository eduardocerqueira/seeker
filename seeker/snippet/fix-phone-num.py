#date: 2025-04-02T16:47:43Z
#url: https://api.github.com/gists/7feea70c9b39d0cbfdf0a46d83e76cab
#owner: https://api.github.com/users/willbeason

import pytest

def fix_phone_num(phone_num_to_fix):
    if len(phone_num_to_fix) != 10:
        raise ValueError(f"phone number must be length 10; got \"{phone_num_to_fix}\" which is of length {len(phone_num_to_fix)}")
    if not phone_num_to_fix.isdigit():
        raise ValueError(f"phone number must only contain digits; got \"{phone_num_to_fix}\"")
    if len(phone_num_to_fix) == 11:
        if phone_num_to_fix[0] != "1":
            raise ValueError(f"phone numbers of length 11 must begin with a 1, got \"{phone_num_to_fix}\"")
        # Strip off the country code "1" from the beginning.
        # Resulting string should be of length 10 to work with formatting code below.
        phone_num_to_fix = phone_num_to_fix[1:]

    # given "5125558823". Split the parts, then recombine and return
    area_code = phone_num_to_fix[0:3]  # 512 (first three digits)
    three_part = phone_num_to_fix[3:6]  # 555 (next three digits)
    four_part = phone_num_to_fix[6:]  # # 8823 (last four digits)

    fixed_num = "(" + area_code + ")" + " " + three_part + " " + four_part

    return fixed_num


def test_fix_phone_num():
    assert fix_phone_num("5125558823") == '(512) 555 8823'

def test_fix_phone_num2():
    assert fix_phone_num("5554429876") == '(555) 442 9876'

def test_fix_phone_num3():
    assert fix_phone_num("3216543333") == '(321) 654 3333'

def test_short_num_error():
  with pytest.raises(ValueError):
    fix_phone_num("51")

def test_long_num_error():
  with pytest.raises(ValueError):
    fix_phone_num("01234567890")

def test_non_digit_error():
  with pytest.raises(ValueError):
    fix_phone_num("012345678a")

def test_fix_phone_num_country_code():
    assert fix_phone_num("15125558823") == '(512) 555 8823'

def test_wrong_country_code():
    with pytest.raises(ValueError):
        assert fix_phone_num("25125558823")
