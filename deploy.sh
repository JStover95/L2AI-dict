echo "Initializing Docker container..."
docker run --name l2ai-mongo -ditp 27017:27017 mongo

if [ $? -ne 0 ]
then
    echo "Docker container failed to initialize."
    exit 1
fi

result=$(python -c "from dictionary import create_db;create_db()" 2>&1)

if [[ "$result" != "True" ]]
then
    echo "Creation of dictionary in MongoDB failed."
    exit 1
else
    echo "Success"
fi
