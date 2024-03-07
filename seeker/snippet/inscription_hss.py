#date: 2024-03-07T16:54:59Z
#url: https://api.github.com/gists/610065052abbe0adaccc95945235e369
#owner: https://api.github.com/users/louisroyer

#!/usr/bin/env python3
#
# Copyright (c) 2024 Louis Royer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import subprocess
import argparse

class Sql:
    def __init__(self, verbose, quiet):
        self.stderr = None
        if not verbose:
            self.stderr = subprocess.DEVNULL
        self.quiet = quiet
        self.MYSQL_USER='root'
        self.MYSQL_PASSWORD= "**********"
        self.base = {
            '001010000000010': {
                'key': '15CB2DAF755D36C72975664210AE7473',
                'opc': '61EB326763D420854677FB6517D5C2BF',
             },
            '001010000000020': {
                'key': '4858C59C3C3ED3BFC7A27EA937DF6942',
                'opc': 'CBB337D6A23418CB730ACD8CDFFE4046',
             },
            '001010000000030': {
                'key': 'EA109CE756C6130513F3766FD9DB0E2B',
                'opc': '69C5356E3C6E5A162F44FF52AA97863C',
             },
            '001010000000030': {
                'key': '760799FD24AB20FFCC41E9A56B429CA3',
                'opc': 'E5690908D2D6FFF1572EECCB1BD646A6',
             },
        }
        self.sql = 'use oai_db; '

    def insert_ue(self, imsi):
        if not imsi in self.base:
            raise KeyError
        self.sql = ''.join([self.sql, 
            '''INSERT INTO `AuthenticationSubscription` (''',
            '''`ueid`, `authenticationMethod`, `encPermanentKey`,''',
            '''`protectionParameterId`, `sequenceNumber`, `authenticationManagementField`,''',
            '''`algorithmId`, `encOpcKey`, `encTopcKey`, `vectorGenerationInHss`, `n5gcAuthMethod`,''',
            f'''`rgAuthenticationInd`, `supi`) VALUES ('{imsi}', '5G_AKA', '{self.base[imsi]['key']}', '{self.base[imsi]['key']}',''',
            ''''{"sqn": "000000000000",''',
            '''"sqnScheme": "NON_TIME_BASED",''',
            '''"lastIndexes": {"ausf": 0}}',''',
            ''''8000',''',
            ''''milenage',''',
            f''''{self.base[imsi]['opc']}',''',
            f'''NULL, NULL, NULL, NULL, '{imsi}');''',
            '''INSERT INTO `SessionManagementSubscriptionData` (`ueid`,''',
            f'''`servingPlmnid`, `singleNssai`, `dnnConfigurations`) VALUES ('{imsi}',''',
            ''''00101', '{"sst": 1, "sd": "1"}','{"oai":{"pduSessionTypes":{''',
            '''"defaultSessionType": "IPV4"},"sscModes": {"defaultSscMode":''',
            '''"SSC_MODE_1"},"5gQosProfile": {"5qi":'''
            '''6,"arp":{"priorityLevel": 1,"preemptCap":''',
            '''"NOT_PREEMPT","preemptVuln":"NOT_PREEMPTABLE"},"priorityLevel":1''',
            '''},"sessionAmbr":{"uplink":"10000Mbps",''',
            '''"downlink":"10000Mbps"},"staticIpAddress":[{"ipv4Addr": "12.1.1.4"}]}}');'''
        ])

    def run(self):
        subprocess.run(['docker', 'exec', 'mysql', 'mysql', f'-u{self.MYSQL_USER}', f'-p{self.MYSQL_PASSWORD}', '-e', self.sql], stderr= "**********"

    def show(self):
        subprocess.run(['docker', 'exec', 'mysql', 'mysql',  f'-u{self.MYSQL_USER}', f'-p{self.MYSQL_PASSWORD}', '-e', 'use oai_db; SELECT * FROM AuthenticationSubscription'], stderr= "**********"

    def reset(self):
        '''This remove all UEs from the database'''
        subprocess.run(['docker', 'exec', 'mysql', 'mysql',  f'-u{self.MYSQL_USER}', f'-p{self.MYSQL_PASSWORD}', '-e', 'use oai_db; truncate AuthenticationSubscription; truncate SessionManagementSubscriptionData'], stderr= "**********"

def __main__():
    parser = argparse.ArgumentParser(prog='inscription_hss', description='Inscription des valeurs correspondante à une SIM dans la HSS')
    parser.add_argument('imsi', help='IMSI correspondant à votre UE')
    parser.add_argument('--quiet', help='Don’t show the content of the database after the insertion', action='store_true')
    parser.add_argument('--reset', help='Clear the content of the database before the insertion', action='store_true')
    parser.add_argument('--verbose', help='Show stderr output', action='store_true')
    args = parser.parse_args()
    try:
        sql = Sql(args.verbose, args.quiet)
        if args.reset:
            sql.reset()
        sql.insert_ue(args.imsi)
        sql.run()
        if not args.quiet:
            sql.show()
    except KeyError:
        print('L’IMSI que vous avez entré est erroné ou n’est pas un de ceux qui a été utilisé lors de la configuration des cartes SIM.')

if __name__ == '__main__':
    __main__()
