#date: 2022-09-21T17:06:20Z
#url: https://api.github.com/gists/ffff5661450ce7d1bc2d814ea494909d
#owner: https://api.github.com/users/MiDemGG

УВВ-111

Вы ландшафтный дизайнер, и вы выполняете крупный заказ! Подсчитайте
стоимость заказа, учитывая стоимость работ (1 кв.м. - 7.61$) и с учетом
финальной скидки 18%. 

baksov = int(input()) * 7.61
print('The final prise is', "%.2f" % (baksov * 0.82), '$')
print('The discount is', "%.2f" % (baksov * 0.18), '$')