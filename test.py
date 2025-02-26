import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mongo"]
collection = db.dysummary
url = "https://stream7.iqilu.com/10339/upload_transcode/202002/09/20200209104902N3v5Vpxuvb.mp4"
document = collection.find_one({"url": url})

# 输出查询结果
if document:
    print("ID:", document.get("_id"))
    print("Summary:", document.get("summary"))
else:
    print("没有找到匹配的文档")
