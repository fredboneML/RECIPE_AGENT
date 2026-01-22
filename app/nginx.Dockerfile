FROM nginx:1.25-alpine

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create directory for SSL certificates
RUN mkdir -p /etc/nginx/ssl

# Create a custom error page for timeouts
RUN mkdir -p /usr/share/nginx/html && \
    echo '<!DOCTYPE html><html><head><title>Processing...</title><meta http-equiv="refresh" content="5"></head><body style="font-family:system-ui;text-align:center;padding:50px;"><h1>Processing your request...</h1><p>The AI is analyzing your query. This may take up to 2 minutes for complex searches.</p><p>The page will automatically refresh. Please wait...</p></body></html>' > /usr/share/nginx/html/50x.html

EXPOSE 80 443

CMD ["nginx", "-g", "daemon off;"]
