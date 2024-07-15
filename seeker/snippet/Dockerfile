#date: 2024-07-15T16:41:00Z
#url: https://api.github.com/gists/a45166a36c2eb652bd7c02e2e5bd4a7b
#owner: https://api.github.com/users/itzzjb

# We will start from a apline version of node.js base image
FROM node:lts-alpine

# Create a directory to hold the application code inside the image
WORKDIR /usr/src/app

# Copy package.json and package-lock.json to the image
COPY ./package*.json /usr/src/app
 
# Installing all the node_modules according to the package.json file
RUN npm install

# Copy the rest of the application code to the image
COPY ./ /usr/src/app

# Expose the port that the app runs on
EXPOSE 3000

# Start the application
CMD ["npm","start"] 