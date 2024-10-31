#date: 2024-10-31T17:05:31Z
#url: https://api.github.com/gists/ea548a58e550f7348f5f82c063b770bf
#owner: https://api.github.com/users/evertonzauso777

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from pymongo import MongoClient
from bson import ObjectId
import os