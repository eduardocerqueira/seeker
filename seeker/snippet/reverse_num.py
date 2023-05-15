#date: 2023-05-15T16:52:53Z
#url: https://api.github.com/gists/31af81596f5d17af666ed5259e437c78
#owner: https://api.github.com/users/depo101

def main():
	given_num = int(input("give me an positive integer number to reverse : "))
	
	num_of_digits = 0
	flag = True
	
	process_1 = given_num
	result = 0
	pow_counter = 0
	
	while flag:
		process_1 = process_1 // 10
		num_of_digits += 1
		if process_1 == 0:
			flag = False
			
			for x in range(num_of_digits-1, -1, -1):
				res = given_num // (10 ** x)
				given_num = given_num - (res * (10 ** x))
				
				result += res * (10 ** pow_counter)
				pow_counter += 1
	
	print(result)		

if __name__ == "__main__":
	main()