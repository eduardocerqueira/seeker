#date: 2022-05-05T16:56:26Z
#url: https://api.github.com/gists/9043eef27f9658005c2407f05546f062
#owner: https://api.github.com/users/neurocean

class APIWrapper:
    
    #snippet...
    
    def poll_api(self, tries, initial_delay, delay, backoff, success_list, apifunction, *args):
        time.sleep(initial_delay)
        for n in range(tries):
            try:
                status = self.get_status()
                if status not in success_list:
                    polling_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
                    print("{0}. Sleeping for {1} seconds.".format(polling_time, delay))
                    time.sleep(delay)
                    delay *= backoff
                else:
                    return apifunction(*args)
            except socket.error as e:
                print("Connection dropped with error code {0}".format(e.errno))
        raise ExceededRetries("Failed to poll {0} within {1} tries.".format(apifunction, tries))