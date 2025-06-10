mkdir gen
protoc --proto_path=. --python_out=gen aruco_message.proto nodes_message.proto
