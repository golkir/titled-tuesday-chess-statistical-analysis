import pymongo
from pymongo import MongoClient
client = MongoClient()
db = client["chess-analysis"]
players = db["players"]


def deleteCollection(collection):
    result = collection.delete_many({})
    # Print information about the deletion
    print(f"Deleted {result.deleted_count} documents from the collection.")


def pushArray(collection, player, array_name, data_to_push):
    # Update the document, pushing the data to the array
    result = collection.update_one(
        {"username": player},  # Specify the document using its identifier
        # Use $push to add data to the array
        {"$push": {array_name: data_to_push}}
    )


def get_element_at_index(collection, username, array_field, game_index):

    # Find the document by ID and retrieve the element at the specified index
    result = collection.find_one({"username": username})
    if result and array_field in result:
        element_at_index = result[array_field][game_index] if game_index < len(
            result[array_field]) else None
    else:
        element_at_index = None
    return element_at_index


def insertIfNotExist(collection, property_name, property_value, player_doc):
    # Check if a document with the specified property name exists
    existing_document = collection.find_one({property_name: property_value})
    # If no document is found, insert a new one
    if existing_document is None:
        new_document = {
            property_name: property_value
        }
        collection.update_one(player_doc,  # Search criteria
                              # Update or insert the document
                              {'$set': new_document},
                              # Set to True for upsert (update or insert)
                              upsert=True
                              )
        print("Document inserted.")
    else:
        print("Document with the specified property name already exists.")


def findAll(collection):
    # Find all documents in the collection
    documents = list(collection.find())
    return documents


def getPlayer(collection, username):
    docs = list(collection.find({"username": username}))
    return docs[0]
