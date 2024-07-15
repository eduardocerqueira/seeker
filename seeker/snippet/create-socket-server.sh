#date: 2024-07-15T17:00:06Z
#url: https://api.github.com/gists/c0553e536ea31f64d3911a5c11dcc572
#owner: https://api.github.com/users/rohitranjan-2702

  #!/bin/bash
  # Create a new directory for the project
  PROJECT_DIR="simple-socket-server"
  mkdir -p $PROJECT_DIR
  cd $PROJECT_DIR

  # Initialize a new Node.js project
  echo "Initializing Node.js project..."
  npm init -y

  # Install necessary packages
  echo "Installing socket.io..."
  npm install socket.io express
  npm install -D nodemon

  # Create a .gitignore
  cat << 'EOF' > .gitignore
  node_modules/
  .env
  .git
  EOF

  # Create a simple socket server file
  cat << 'EOF' > index.js
  const express = require("express");
  const { createServer } = require("node:http");
  const { Server } = require("socket.io"); 

  const app = express(); 
  const server = createServer(app); 
  const io = new Server(server); 

  app.get("/", (req, res) => {
    res.send("<h1>Socket Server is running 🚀</h1>");
  });

  io.on("connection", (socket) => {
    console.log("User connected", socket.id);

    socket.on("message", (message) => {
      const msg = JSON.parse(message);
      console.log(msg);
    });

    socket.on("disconnect", () => {
      console.log("User Disconnected", socket.id);
    });
  });

  server.listen(3000, () => {
    console.log("server running at http://localhost:3000");
  });

  EOF

  # Add scripts to package.json
  echo "Adding dev and start scripts to package.json..."
  node -e "
  const fs = require('fs');
  const packageJson = JSON.parse(fs.readFileSync('package.json'));
  packageJson.scripts = {
    ...packageJson.scripts,
    dev: 'nodemon index.js',
    start: 'node index.js'
  };
  fs.writeFileSync('package.json', JSON.stringify(packageJson, null, 2));
  "

  # Create a .gitignore
  cat << 'EOF' > READme.md
  # Simple Socket Server

  ## Setup

  - `cd simple-socket-server`
  - `npm run dev` : for development mode
  - `npm start` : for production mode
  EOF

  # Instructions to run the server
  echo "Setup complete. To run the server, execute the following commands:"
  echo "cd $PROJECT_DIR"
  echo "npm run dev   # For development mode"
  echo "npm start     # For production mode"
