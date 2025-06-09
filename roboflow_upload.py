from roboflow import Roboflow

# Initialize with your API key
rf = Roboflow(api_key="WCNk9uoVTJzAEs4S3Waq")

# Connect to your workspace and project
project = rf.workspace("debrajm").project("dldv2")
project.upload(r"C:\Users\debra\Desktop\CODE\Dataset\Drive_Dataset_Annotated\train.zip")