# Frontend Dockerfile
FROM node:16-alpine

# Install necessary packages
RUN apk add --no-cache shadow

# Set working directory
WORKDIR /usr/src/app

# Define build arguments for UID and GID
ARG USER_ID
ARG GROUP_ID

# Create user and group more safely
RUN if [ -z "$GROUP_ID" ]; then \
      GROUP_ID=1000; \
    fi && \
    if [ -z "$USER_ID" ]; then \
      USER_ID=1000; \
    fi && \
    # First try to create the group
    if ! getent group appgroup > /dev/null 2>&1; then \
      addgroup -g $GROUP_ID -S appgroup || addgroup -S appgroup; \
    fi && \
    # Then create the user
    adduser -u $USER_ID -S appuser -G appgroup -h /home/appuser

# Setup npm directories
RUN mkdir -p /home/appuser/.npm /home/appuser/.npm-global && \
    chown -R appuser:appgroup /home/appuser/.npm /home/appuser/.npm-global

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