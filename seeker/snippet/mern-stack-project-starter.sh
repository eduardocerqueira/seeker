#date: 2023-03-08T17:09:55Z
#url: https://api.github.com/gists/4effac7c74884659810c053b336d5df5
#owner: https://api.github.com/users/kapilbhise

#!/bin/bash

echo "Setting up a new MERN project..."

# Get project name from user
read -p "Enter project name: " project_name

# Create project directory
mkdir $project_name
cd $project_name

# Initialize npm and create package.json
npm init -y

# Install dependencies
npm install express mongoose concurrently

# Install dev dependencies
npm install nodemon --save-dev

# Create server.js file
mkdir db
mkdir model
mkdir router

touch index.js
echo "const express = require('express');" >> server.js
echo "const app = express();" >> server.js
echo "const mongoose = require('mongoose');" >> server.js
echo "" >> server.js
echo "// Connect to MongoDB" >> server.js
echo "mongoose.connect('mongodb://localhost/${project_name}');" >> server.js
echo "" >> server.js
echo "// Start server" >> server.js
echo "const port = process.env.PORT || 5000;" >> server.js
echo "app.listen(port, () => {" >> server.js
echo "  console.log(\`Server started on port \${port}\`);" >> server.js
echo "});" >> server.js

# Create client directory
mkdir client
cd client

# Initialize create-react-app and create package.json
npx create-react-app .
rm src/index.js
touch src/index.js
echo "import React from 'react';" >> src/index.js
echo "import ReactDOM from 'react-dom';" >> src/index.js
echo "" >> src/index.js
echo "ReactDOM.render(<h1>Hello, world!</h1>, document.getElementById('root'));" >> src/index.js

# Create .env file for client
touch .env
echo "PORT=3000" >> .env

# Go back to project root directory
cd ..

# Create .gitignore file in server directory
touch .gitignore
echo "node_modules/" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "client/node_modules/" >> .gitignore

# Remove .gitignore file from client directory
rm client/.gitignore

# Update scripts object in package.json
npm install -g json

npx json -I -f package.json -e 'this.scripts = {"start": "cd client && npm start", "server": "nodemon server.js", "dev": "concurrently --kill-others-on-fail \"npm run server\" \"npm run start\""}'


echo "Setup complete!"