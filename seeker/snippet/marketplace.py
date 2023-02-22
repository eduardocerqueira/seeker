#date: 2023-02-22T17:03:03Z
#url: https://api.github.com/gists/596c202d6ca8877e40a839273f80ae4d
#owner: https://api.github.com/users/Varsha-R

import heapq

class Marketplace:
    def __init__(self, buyOffers, sellOffers):
        self.buyOffers = [-offer for offer in buyOffers]
        heapq.heapify(self.buyOffers)
        self.sellOffers = [offer for offer in sellOffers]
        heapq.heapify(self.sellOffers)
        print(self.buyOffers, self.sellOffers)
    
    def matchNewBuyOffer(self, price):
        if len(self.sellOffers) == 0 or price < self.sellOffers[0]:
            print("No match!")
            heapq.heappush(self.buyOffers, -1*price)
        matchedOffer = heapq.heappop(self.sellOffers)
        return matchedOffer
    
    def matchNewSellOffer(self, price):
        if len(self.buyOffers) == 0 or price > -1 * self.buyOffers[0]:
            print("No match!")
            heapq.heappush(self.sellOffers, price)
        matchedOffer = -1 * heapq.heappop(self.buyOffers)
        return matchedOffer

marketplace = Marketplace(buyOffers=[90, 99, 99, 100, 100], sellOffers=[110, 110, 120])
print(marketplace.matchNewSellOffer(95)) # matches to $100
print(marketplace.matchNewBuyOffer(150)) # matches to $110
marketplace.matchNewBuyOffer(109) # no match, added to heap marketplace.match_new_sell_offer(100) # matches to $109)
print(marketplace.buyOffers)
print(marketplace.sellOffers)

# Reference - https://www.bartleby.com/questions-and-answers/revise-a-marketplace-class-in-python-that-allows-users-to-buysell-stocks.-it-will-match-buyers-with-/ba1cb6bb-0701-4616-802f-7fd84e5c4d94 