#date: 2022-03-14T17:05:38Z
#url: https://api.github.com/gists/cb7b5772c08a7a9c355eb04c597cd8ca
#owner: https://api.github.com/users/anthonymiyoro

import requests, json, pprint, googlemaps

def calculate_ride_time_and_distance(source_string,dest_string):
    
    # Requires API key
    gmaps = googlemaps.Client(key='API_KEY_HERe')
    
    # Requires cities name
    my_dist = gmaps.distance_matrix(source_string,dest_string)['rows'][0]['elements'][0]
    # {'distance': {'text': '2.3 km', 'value': 2327},
    # 'duration': {'text': '7 mins', 'value': 449},
    # 'status': 'OK'}
    
    # print the value of x
    pprint.pprint(my_dist)
    
    for key, value in my_dist.items():
        if key == 'distance':
            distance_value = value['value']
        if key == 'duration':
            duration_value = value['value']
            
    converted_distance = distance_value / 1000
    converted_duration = duration_value / 60
    
    calculate_ride_cost(converted_distance, converted_duration)
    return ("Done")
                
def calculate_ride_cost(time_of_ride, ride_distance):
    base_fare = 100.00
    cost_per_minute = 0.00
    cost_per_kilometre = 0.03
    booking_fee = 20.00
    
    Fare = (base_fare + ((cost_per_minute * time_of_ride) + (cost_per_kilometre * ride_distance)) + booking_fee 

    print ("Fare: ",Fare)
    return Fare

source_string = "Yaya Center, Nairobi"
dest_string = "Valley Arcade, Nairobi"
calculate_ride_time_and_distance(source_string,dest_string)