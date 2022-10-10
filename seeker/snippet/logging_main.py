#date: 2022-10-10T17:06:11Z
#url: https://api.github.com/gists/379bf88c63abc2e359b6f6072a5ee7d2
#owner: https://api.github.com/users/jongphago

if __name__ == "__main__":
    logger = logging.getLogger(__name__)  # get logger from parent module
    logging.Formatter.converter = timetz  #
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%d/%b/%Y %H:%M:%S',
                        level=logging.DEBUG)
    print("Excute logging.py")