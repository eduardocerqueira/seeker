#date: 2024-09-30T17:01:19Z
#url: https://api.github.com/gists/c206544b64eadf1ec4e2e0ce08f37887
#owner: https://api.github.com/users/manyajsingh

from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson import ObjectId

app = Flask(__name__)

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/jalveksha_db"
mongo = PyMongo(app)

@app.route('/api/water_quality', methods=['POST'])
def add_water_quality_data():
    data = request.json
    water_quality = {
        "location": data["location"],
        "date": data["date"],
        "dissolved_oxygen": data["dissolved_oxygen"],
        "biodiversity_index": data["biodiversity_index"],
        "nitrate_level": data["nitrate_level"],
        "phosphorus_level": data["phosphorus_level"],
        "status": data["status"]
    }
    mongo.db.water_quality.insert_one(water_quality)
    return jsonify({"message": "Water quality data added successfully!"}), 201

@app.route('/api/water_quality', methods=['GET'])
def get_water_quality_data():
    water_quality_data = mongo.db.water_quality.find()
    result = []
    for data in water_quality_data:
        data["_id"] = str(data["_id"])  # Convert ObjectId to string
        result.append(data)
    return jsonify(result), 200

@app.route('/api/water_quality/<id>', methods=['GET'])
def get_single_water_quality_data(id):
    data = mongo.db.water_quality.find_one({"_id": ObjectId(id)})
    if data:
        data["_id"] = str(data["_id"])  # Convert ObjectId to string
        return jsonify(data), 200
    return jsonify({"message": "Data not found!"}), 404

@app.route('/api/water_quality/<id>', methods=['PUT'])
def update_water_quality_data(id):
    data = request.json
    updated_data = {
        "location": data["location"],
        "date": data["date"],
        "dissolved_oxygen": data["dissolved_oxygen"],
        "biodiversity_index": data["biodiversity_index"],
        "nitrate_level": data["nitrate_level"],
        "phosphorus_level": data["phosphorus_level"],
        "status": data["status"]
    }
    result = mongo.db.water_quality.update_one({"_id": ObjectId(id)}, {"$set": updated_data})
    if result.modified_count > 0:
        return jsonify({"message": "Water quality data updated successfully!"}), 200
    return jsonify({"message": "No changes made or data not found!"}), 404

@app.route('/api/water_quality/<id>', methods=['DELETE'])
def delete_water_quality_data(id):
    result = mongo.db.water_quality.delete_one({"_id": ObjectId(id)})
    if result.deleted_count > 0:
        return jsonify({"message": "Water quality data deleted successfully!"}), 200
    return jsonify({"message": "Data not found!"}), 404

if __name__ == '__main__':
    app.run(debug=True)
