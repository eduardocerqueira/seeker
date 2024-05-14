#date: 2024-05-14T17:04:50Z
#url: https://api.github.com/gists/e45a7fa2c7f5b981e2fef47f431eb410
#owner: https://api.github.com/users/rz3n

#!/usr/bin/env python3

# Simple python script to manage my Cloudflare Email routes via API
# API Documentation: https://developers.cloudflare.com/api/operations/email-routing-routing-rules-list-routing-rules

import requests
import sys

CF_API_TOKEN = "**********"
CF_ZONE_ID = "YOUR_CF_ZONE_ID"
BIN_NAME = sys.argv[0]
MAIN_DOMAIN = "YOUR_DOMAIN"
RANDOM_STRING_SIZE = 7


# Help message
def display_help():
  print("Usage:", BIN_NAME, " [-l | --list] [-a | --add] [-u | --update] [-d | --delete]")
  print("  -l, --list")
  print("    List all Cloudflare Email Routes.")
  print("  -a, --add <alias> <destination> <name>")
  print("    Add a new Cloudflare Email Route.")
  print("    If <alias> is 'random', a random string will be generated.")
  print("    Name is optional and is like a description of the rule.")
  print("  -u, --update <tag> <destination>")
  print("    Update an existing Cloudflare Email Route.")
  print("  -d, --delete <tag>")
  print("    Delete an existing Cloudflare Email Route.")
  print("  -h, --help")
  print("    Display this help message.")
  print("\n")
  sys.exit(1)


# Function to get a route by tag
def get_route(tag):
  routes = get_routes()
  for route in routes['result']:
    if route.get('tag') == tag:
      return route


# Function to get all routes
def get_routes():
  url = 'https://api.cloudflare.com/client/v4/zones/' + CF_ZONE_ID + '/email/routing/rules'
  headers = {'Authorization': "**********"
  r = requests.get(url, headers=headers)
  return r.json() 


# Function to show all routes
def show_routes():
  routes = get_routes()
  for route in routes['result']:
    name = route.get('name')
    tag = route.get('tag')
    matcher_value = route['matchers'][0].get('value')
    action_type = route['actions'][0].get('type')
    action_value = route['actions'][0].get('value')
    enabled = route.get('enabled')

    # Print the routes
    if route == routes['result'][0]:
      print("\tName\t\t\tMatcher\t\tAction Type\t\tAction Value\t\tEnabled\t\tTag")
    print("  - ", name, "\t", matcher_value, "\t", action_type, "\t", action_value, "\t", enabled, "\t", tag)


# Function to add a new route
def add_route(alias, destination, name):
  url = f'https://api.cloudflare.com/client/v4/zones/{CF_ZONE_ID}/email/routing/rules'
  headers = {'Authorization': "**********": 'application/json'}
  payload = {
    "actions": [
      {
        "type": "forward",
        "value": [
          destination
        ]
      }
    ],
    "enabled": True,
    "matchers": [
      {
        "field": "to",
        "type": "literal",
        "value": alias
      }
    ],
    "name": name,
    "priority": 0
  }
  response = requests.post(url, headers=headers, json=payload)
  if response.status_code == 200:
    print("Route added successfully!")
  else:
    print("Failed to add route. Error:", response.json())


# Function to update the route destination by tag
def update_route(tag, destination):
  route = get_route(tag)
  alias = route['matchers'][0].get('value')

  url = f'https://api.cloudflare.com/client/v4/zones/{CF_ZONE_ID}/email/routing/rules/{tag}'
  headers = {'Authorization': "**********": 'application/json'}
  payload = {
    "actions": [
      {
        "type": "forward",
        "value": [
          destination
        ]
      }
    ],
    "enabled": True,
    "matchers": [
      {
        "field": "to",
        "type": "literal",
        "value": alias
      }
    ],
    "name": f"Send to {destination} rule.",
    "priority": 0
  }
  response = requests.put(url, headers=headers, json=payload)
  if response.status_code == 200:
    print("Route updated successfully!")
  else:
    print("Failed to update route. Error:", response.json())


# Function to delete a route by tag
def delete_route(tag):
  url = f'https://api.cloudflare.com/client/v4/zones/{CF_ZONE_ID}/email/routing/rules/{tag}'
  headers = {'Authorization': "**********": 'application/json'}
  response = requests.delete(url, headers=headers)
  if response.status_code == 200:
    print("Route deleted successfully!")
  else:
    print("Failed to delete route. Error:", response.json())


# Function to generate a random string of length 'length'
def random_string(length):
  import random
  import string
  return ''.join(random.choice(string.ascii_lowercase) for i in range(length))



if __name__ == '__main__':
  
  # Help statement
  if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help']:
    display_help()

  # List all routes
  elif sys.argv[1] in ['-l', '--list']:
    print("All Cloudflare Email Routes:")
    show_routes()

  # Add a new route
  elif sys.argv[1] in ['-a', '--add']:
    if len(sys.argv) != 5:
      print("Usage:", BIN_NAME, " -a <alias> <destination> <name>")
      sys.exit(1)
    else:
      if sys.argv[2] == 'random':
        alias = random_string(RANDOM_STRING_SIZE) + "@" + MAIN_DOMAIN
      else:
        alias = sys.argv[2] + "@" + MAIN_DOMAIN
      print("Adding route with alias", alias, "to", sys.argv[3])
      add_route(alias, sys.argv[3], sys.argv[4])

  # Update an existing route
  elif sys.argv[1] in ['-u', '--update']:
    if len(sys.argv) != 4:
      print("Usage:", BIN_NAME, " -u <tag> <destination>")
      sys.exit(1)
    else:
      print("Updating route with tag", sys.argv[2], "to", sys.argv[3])
      update_route(sys.argv[2], sys.argv[3])

  # Delete an existing route
  elif sys.argv[1] in ['-d', '--delete']:
    if len(sys.argv) != 3:
      print("Usage:", BIN_NAME, " -d <tag>")
      sys.exit(1)
    else:
      print("Deleting route with tag", sys.argv[2])
      delete_route(sys.argv[2])
s.exit(1)
    else:
      print("Deleting route with tag", sys.argv[2])
      delete_route(sys.argv[2])
