FROM node:16-alpine

# Install necessary packages
RUN apk add --no-cache shadow

# Set working directory
WORKDIR /usr/src/app

# Define build arguments for UID and GID
ARG USER_ID=1000
ARG GROUP_ID=1000

# Setup user and group with checks for existing ones
RUN \
    # Check if group exists and use it, otherwise create new
    if getent group ${GROUP_ID} > /dev/null 2>&1; then \
    existing_group=$(getent group ${GROUP_ID} | cut -d: -f1) && \
    echo "Group ${GROUP_ID} exists as ${existing_group}, using it" ; \
    else \
    addgroup -g ${GROUP_ID} -S appgroup && \
    echo "Created new group appgroup with ID ${GROUP_ID}" && \
    existing_group="appgroup" ; \
    fi && \
    # Check if user exists and use it, otherwise create new
    if getent passwd ${USER_ID} > /dev/null 2>&1; then \
    existing_user=$(getent passwd ${USER_ID} | cut -d: -f1) && \
    echo "User ${USER_ID} exists as ${existing_user}, using it" ; \
    else \
    adduser -u ${USER_ID} -S appuser -G ${existing_group} -h /home/appuser && \
    echo "Created new user appuser with ID ${USER_ID}" && \
    existing_user="appuser" ; \
    fi && \
    # Create necessary directories regardless of user/group scenario
    mkdir -p /home/${existing_user}/.npm /home/${existing_user}/.npm-global && \
    chown -R ${USER_ID}:${GROUP_ID} /home/${existing_user}

# Set npm configuration
ENV NPM_CONFIG_PREFIX=/home/appuser/.npm-global
ENV PATH="/home/appuser/.npm-global/bin:$PATH"
ENV NPM_CONFIG_CACHE=/home/appuser/.npm

# Copy package files
COPY package*.json ./

# Set permissions and install dependencies as root, then fix permissions
RUN chown -R ${USER_ID}:${GROUP_ID} /usr/src/app && \
    mkdir -p /home/appuser/.npm && \
    chown -R ${USER_ID}:${GROUP_ID} /home/appuser/.npm && \
    npm install && \
    mkdir -p node_modules/.cache && \
    chmod -R 777 node_modules/.cache && \
    chown -R ${USER_ID}:${GROUP_ID} /usr/src/app

# Copy the rest of the application
COPY --chown=${USER_ID}:${GROUP_ID} . .

# Switch to appropriate user
USER ${USER_ID}:${GROUP_ID}

EXPOSE 3000

# Start the application
CMD ["npm", "start"]