FROM node:16-alpine

# Install necessary packages
RUN apk add --no-cache shadow

# Set working directory
WORKDIR /usr/src/app

# Define build arguments for UID and GID
ARG USER_ID
ARG GROUP_ID

# Create a non-root user with the provided UID and GID
RUN groupadd -g ${GROUP_ID:-1000} appgroup && \
    useradd -u ${USER_ID:-1000} -g appgroup -m appuser

# Copy package files
COPY package*.json ./

# Set ownership and install dependencies
RUN chown -R appuser:appgroup /usr/src/app && \
    npm install && \
    mkdir -p node_modules/.cache && \
    chmod -R 777 node_modules/.cache

# Copy the rest of the application
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

EXPOSE 3000

# Start the application
CMD ["npm", "start"]
