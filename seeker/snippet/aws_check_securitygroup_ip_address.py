#date: 2021-12-10T17:16:06Z
#url: https://api.github.com/gists/37fcfe3f0297eca214e2374f637e122c
#owner: https://api.github.com/users/shollingsworth

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show sg."""
import argparse
import ipaddress
import boto3


def get_rules(args):
    """Return list of CIDRs for a security group."""
    sess = boto3.Session(
        profile_name=args.profile,
        region_name=args.region,
    )
    ec2 = sess.client("ec2")
    response = ec2.describe_security_groups(GroupIds=[args.sg])
    for i in response["SecurityGroups"]:
        for z in i["IpPermissions"]:  # type: ignore
            for y in z["IpRanges"]:  # type: ignore
                yield (
                    z["IpProtocol"],  # type: ignore
                    int(z["FromPort"]),  # type: ignore
                    int(z["ToPort"]),  # type: ignore
                    y["CidrIp"],  # type: ignore
                )


def in_proto(args, proto):
    if not args.proto:
        return True
    return proto == args.proto


def in_port_range(args, fport, tport):
    if not args.port:
        return True
    return fport <= args.port <= tport


def main(args):
    """Run main function."""
    rules = list(get_rules(args))
    for ip in args.ipaddress:
        ip = ipaddress.ip_address(ip)
        _ip_match = 0
        _proto_match = 0
        _port_match = 0
        for proto, fport, tport, cidr in rules:
            cidr = ipaddress.ip_network(cidr)
            _ip_match = max(_ip_match, ip in cidr)
            _proto_match = max(_proto_match, int(in_proto(args, proto)))
            _port_match = max(_port_match, int(in_port_range(args, fport, tport)))
            if all([_ip_match, _proto_match, _port_match]):
                break
        match_dict = {
            "Ip Match": bool(_ip_match),
            "Proto Match": bool(_proto_match),
            "Port Match": bool(_port_match),
        }

        if all(match_dict.values()):
            tarr = ["MATCH"]
        else:
            tarr = ["MISS"]
        tarr.append(str(ip))
        tarr.extend([f"""{k}: {v}""" for k, v in match_dict.items()])
        print("\t".join(tarr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "-p",
        "--profile",
        help="aws profile",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-r",
        "--region",
        help="aws region",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--sg",
        help="security group id",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--port",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--proto",
        type=str,
        default=None,
    )
    parser.add_argument(
        "ipaddress",
        nargs="+",
        type=str,
    )
    # main_args = parser.parse_args(['--sg', 'sg-foo', '1', '2', '3'])
    main_args = parser.parse_args()
    main(main_args)
