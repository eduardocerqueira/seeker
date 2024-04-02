#date: 2024-04-02T16:54:19Z
#url: https://api.github.com/gists/280ba034b2eb07aac455db6407219487
#owner: https://api.github.com/users/Dhanasaitholeti

# making a folder for server and installing the main dependecies
mkdir server
cd server
yarn init -y
yarn add express cors dotenv body-parser mongoose
yarn add -D @types/node @types/express @types/cors @types/dotenv @types/body-parser typescript ts-node nodemon

# settign up .gitignore
touch .gitignore
echo "node_modules" >> .gitignore
echo ".env" >> .gitignore

# setting up the typescript
touch tsconfig.json
echo '{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "baseUrl": "src",
    "outDir": "dist",
    "sourceMap": false,
    "noImplicitAny": true
  },
  "include": ["src/**/*"]
}
' >> tsconfig.json

# setting up nodemon.
touch nodemon.json
echo '{
  "watch": ["src"],
  "ext": ".ts,.js",
  "exec": "ts-node ./src/index.ts"
}' >> nodemon.json


# setting up folders and basic workflow
mkdir src
cd src
mkdir controllers models routes utils  middlewares
touch index.ts
echo '
import cors from "cors";
import bodyParser from "body-parser";
import express, { Application } from "express";

import { RouteHandler } from "./routes";
import ErrorMiddleware from "./middlewares/error.middleware";

const app: Application = express();

app.use(cors());
app.use(express.json());
app.use(bodyParser.json());
app.use(express.urlencoded());

RouteHandler(app); //for handling routes.

app.use(ErrorMiddleware);

app.listen(8000, () => {
  console.log("The server is running on http://localhost:8000/");
});
' >> index.ts

touch middlewares/error.middleware.ts
echo '
import { NextFunction, Request, Response } from "express";

const ErrorMiddleware = (
  err: Error,
  _req: Request,
  res: Response,
  _next: NextFunction
) => {
  res.json({ error: true, message: err.message });
};

export default ErrorMiddleware;
" >> middlewares/error.middleware.ts

touch routes/index.ts
echo "
import { Application, Request, Response } from "express";

export const RouteHandler = (app: Application) => {
  app.get("/", (req: Request, res: Response) => {
    res.status(200).send("The server is working");
  });
};
' >> routes/index.ts

