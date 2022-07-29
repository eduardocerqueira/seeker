#date: 2022-07-29T16:49:39Z
#url: https://api.github.com/gists/2e588c439024bef382cb6958982eaf4e
#owner: https://api.github.com/users/damullutkid

courses = ['business', 'compsci', 'markiting', 'managment']
course_num = len(courses)
print(courses)
print('number of courses:', + course_num)
add = input('\n would you like to add or remove courses?, to add type add, to remove type remove, no = no action: ')
add.lower()

if add == 'add':
    add_course = input('What course would you like to add?')
    courses.append(add_course.lower())
    print(courses)
elif add == 'remove':
    remove = input('what course would you like to remove?')
    print('\n removing, ', remove)
    courses.remove(remove.lower())
elif add == 'no':
    print('courses have been set:' '\n ', courses)

print('\nHere are your courses written in aplhbitical order: \n', sorted(courses))

print('\n list as stored in program: \n', courses)

