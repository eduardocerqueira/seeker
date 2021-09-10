#date: 2021-09-10T16:52:07Z
#url: https://api.github.com/gists/62ecdaace28a62ae7f290ceb1d89e9db
#owner: https://api.github.com/users/tuian

#!/usr/bin/python
import requests
from socket import *
from requests.packages.urllib3.contrib import pyopenssl as reqs
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import argparse
import ipaddress
#import asyncio

#global variable to store results because yolo
results = {}

def print_banner():
    print(r"""
                                                                                
                                                                                
                                                                                
                                                              *,,,,,,,*         
                                                          .,,,,,,,.,,,/*        
                                                       *,,,,,,,,,,,*((%.        
                                                   .*,,,,..,,,,*(#%%%/,         
                                                /,,,,,,,,,,((#%&%(*,.           
                                             (,,,..,.,,((#%%#/*,.               
                               .............,%##,,*((#%%#/*,                    
                    ,*,,,,,,,,,,,.,,,..........#(#%%(/*,                        
            .*,,,*,,,,,,,,,,,,,,,,,,,,,,,,......(/,,                            
       . ..,,,,,,,,,,,,,,,,,,,,,,,,,,,,........*.                               
     ,/(*  .,,,,,,,,,,,,,*,,,,,,,,,,,,,,..... ,                                 
      .*(#*  .,,,,,,,,,,,,,,,,,,,,,,,,,,....*,                                  
        .*(#,  .,,,,,,,,,,,,,,,,,,,,.,,,...*                                    
          ,*(#,  .,,,,,,,,,,,,,,,,,,,....*,                                     
            .*(#,  .,,,,,,,,,,,,,,,,,.,*.                                       
              .*(#.  .,,,,,,,,,,,,,.(*.                                         
                ,*(#.  ..,,,,,,,.*/,.                                           
                  ,*(#.  ,,,..*/*.                                              
                    ,/(#   #/*.                                                 
                      ,*/*,                                                     
                                                              

    SSL Hostname Scraper - jfmaes

    """)

def main():
    print_banner()
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip",help="the ip address (or range) you want to extract the hostnames from",required=False)
    parser.add_argument("-f","--file",help="file containing targets to extract hostnames from",required=False)
    parser.add_argument("-o","--output",help="where to store the output",required=False)
    parser.add_argument("-t","--time-out",help="the max time out you want to wait to retreive SSL certs (default 3 sec)",required=None,default=3,type=int)
    parser.add_argument("-v","--verbose",help="enable verbose output",required=False,action='store_true',default=False)
    parser.add_argument("-vv","--very-verbose",help="shows you even more BS on your terminal!",action='store_true',required=False,default=False)

    args = parser.parse_args()
    if not args.ip and not args.file:
        raise Exception("you need either an ip, an ip range or a target file!")

    #populate results
    results = extract_hostnames(args.ip,args.file,args.verbose,args.time_out,args.very_verbose)

    #write results
    if args.output:
        write_results_to_file(results,args.output)
    if args.verbose or args.very_verbose or not args.output:
        write_results_to_console(results)
    
    print("[*] Finished Scraping hostnames! Enjoy")



def extract_hostnames(ip,target_list,verbose=False,timeout=3,very_verbose=False):
    if ip:
        if "/" in ip:
            print("[*] Your target appears to be an IP Range, extracting all IPs from {0} ...".format(ip))
            ip_addressess = [str(ip) for ip in ipaddress.IPv4Network(ip)]
            print("[*] extracting hostnames from the SSL cert(s)")
            #slicing the list as we dont want the network address or the broadcast address
            for ip_address in ip_addressess:
                try:
                    extract_hostname_ssl(ip_address,verbose,timeout,very_verbose)
                except Exception:
                    if very_verbose:
                        print("[!] something went wrong with extracting the cert for {0}, probably a connection timeout..\n".format(ip_address))
                    pass
        else:
            print("[*] extracting hostnames from the SSL cert")
            extract_hostname_ssl(ip,verbose,timeout,very_verbose)
    elif target_list:
        infile = open(target_list,'r')
        for line in infile.readlines():
            #recursion wtfffff :D 
            extract_hostnames(line.rstrip('\n'),None,verbose,timeout)
    return results


def extract_hostname_ssl(ip,verbose,timeout,very_verbose):
    if very_verbose:
        print("[*] now inspecting {0} ... \n".format(ip))
    hostnames = []
    try:
        setdefaulttimeout(timeout)
        x509 = reqs.OpenSSL.crypto.load_certificate(
           reqs.OpenSSL.crypto.FILETYPE_PEM,
          reqs.ssl.get_server_certificate((ip, 443))
          )
    except Exception as e:
        raise e
    alt_names = reqs.get_subj_alt_name(x509)
    for tuple in alt_names:
        recordtype,recordvalue = tuple
        if "DNS" in recordtype:
            if recordvalue not in hostnames:
                hostnames.append(recordvalue)
    hostnames_string = ", ".join(hostnames)
    if verbose:
        print("[+] {0} : {1}\n" .format(ip,hostnames_string))
    results[ip] = hostnames_string
    if very_verbose:
        print("[*] Results contains: {0} \n".format(results))
    

def write_results_to_file(results,outfile):
    outfile = open(outfile,'a')
    for k,v in results.items():
            outfile.writelines("{0} : {1}".format(k,v))
    outfile.close()
        
def write_results_to_console(results):
    print("[*] Here are your results! \n")
    for k,v in results.items():
        print("{0} : {1}".format(k,v))
    
if __name__ == "__main__":
    main()