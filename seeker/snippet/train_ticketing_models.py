#date: 2021-11-30T17:12:01Z
#url: https://api.github.com/gists/19dd038ab5ebb8d264a6e345bb3940c5
#owner: https://api.github.com/users/efojs

from django.db import models


class Train(models.Model):
    name = models.CharField(max_length=128)


class Car(models.Model):
    seats_range = models.PositiveSmallIntegerField()
    next = models.OneToOneField(
        "self", related_name="previous", on_delete=models.CASCADE
    )
    train = models.ForeignKey(Train, on_delete=models.CASCADE)


class Station(models.Model):
    name = models.CharField(max_length=256)


class Stop(models.Model):
    arrival_at = models.DateTimeField()
    departure_at = models.DateTimeField()
    station = models.ForeignKey(Station, on_delete=models.CASCADE)
    next = models.OneToOneField(
        "self", related_name="previous", on_delete=models.CASCADE
    )
    train = models.ForeignKey(Train, on_delete=models.CASCADE)


class Passenger(models.Model):
    first_name = models.CharField(max_length=128)
    last_name = models.CharField(max_length=128)
    passport = models.CharField(max_length=128)


class Ticket(models.Model):
    seat = models.PositiveSmallIntegerField()
    price = models.DecimalField(max_digits=8, decimal_places=2)
    car = models.ForeignKey(Car, on_delete=models.SET_NULL, null=True)
    departure = models.ForeignKey(
        Stop, related_name="departures", on_delete=models.SET_NULL, null=True
    )
    destination = models.ForeignKey(
        Stop, related_name="arrivals", on_delete=models.SET_NULL, null=True
    )
    passenger = models.ForeignKey(Passenger, on_delete=models.SET_NULL, null=True)
