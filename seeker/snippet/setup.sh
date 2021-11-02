#date: 2021-11-02T17:11:36Z
#url: https://api.github.com/gists/e49e919ebc464c01397605e9d20b25bb
#owner: https://api.github.com/users/aadityarajkumawat

#!/bin/bash

# setup frontend
git clone https://github.com/shreya250101/artistry_front-end.git frontend
cd frontend
yarn

# go back to root
cd ..

# setup backend
# TODO: put the real-github-repo
git clone https://github.com/aadityarajkumawat/artistry-backend.git backend
cd backend

# install dependencies
yarn
# create environment file
touch .env

# add environment variables
echo "PORT=4001" >> .env
echo "COOKIE_SECRET=dhbfkjdvfasdfveuwfvwjdfhvdsjfvha" >> .env
echo "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/artistry_dev?schema=public" >> .env

# setup prisma
npx prisma generate
npx prisma migrate dev --name setup

# build
yarn build
cp -r ./src/graphql ./dist/
