# Dockerfile for frontend_app
FROM node:16-alpine

# Set working directory
WORKDIR /usr/src/app

# Copy package.json and package-lock.json (if present)
COPY package.json package-lock.json ./

# Install dependencies with npm
RUN npm install

# Copy the rest of the application
COPY . .

# Build the application
RUN npm run build

# Expose the port and start the app
EXPOSE 3000
CMD ["npm", "start"]
