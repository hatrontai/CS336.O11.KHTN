from database_manager import database_manager
import redis

# This is a dummy file to test query function

r = redis.Redis()

db = database_manager()
res = db.query('hello')

for doc in res:
    print(doc['dir'])
