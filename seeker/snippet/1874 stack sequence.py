#date: 2025-07-15T16:59:59Z
#url: https://api.github.com/gists/f491b076c6bf82ef317d8ff47c99eefb
#owner: https://api.github.com/users/Gloveman

def solve_stack_problem():
    """
    1부터 n까지의 수를 스택에 오름차순으로 push, pop하여
    주어진 수열을 만들 수 있는지 확인하는 함수.
    """
    try:
        n = int(sys.stdin.readline())
        goal = [int(sys.stdin.readline()) for _ in range(n)]
    except (IOError, ValueError) as e:
        print(f"입력 처리 중 오류 발생: {e}")
        return

    stack = []
    answer = []
    current_num = 1  # 스택에 push할 1부터 시작하는 오름차순 숫자
    possible = True

    for target_num in goal:
        # target_num을 만들 때까지 스택에 push
        while current_num <= target_num:
            stack.append(current_num)
            answer.append('+')
            current_num += 1

        # 스택의 맨 위가 target_num과 일치하면 pop
        if stack and stack[-1] == target_num:
            stack.pop()
            answer.append('-')
        else:
            # 일치하지 않으면 이 수열은 만들 수 없음
            possible = False
            break

    if possible:
        for op in answer:
            print(op)
    else:
        print('NO')