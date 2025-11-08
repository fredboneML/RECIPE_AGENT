import { jwtDecode } from 'jwt-decode';

const TOKEN_KEY = 'token';
const TENANT_CODE_KEY = 'tenantCode';

class TokenManager {
  constructor() {
    this.refreshTimeout = null;
    this.refreshThreshold = 5 * 60 * 1000; // 5 minutes before expiry
    this.isRefreshing = false;
    this.subscribers = [];
  }

  // Get backend URL dynamically
  getBackendUrl() {
    return window.location.hostname === 'localhost' 
      ? 'http://localhost:8000'
      : `http://${window.location.hostname}:8000`;
  }

  getToken() {
    return localStorage.getItem(TOKEN_KEY);
  }

  setToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
    this.scheduleRefresh(); // Auto-schedule refresh when setting token
  }

  clearToken() {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(TENANT_CODE_KEY);
    localStorage.removeItem('userName');
    localStorage.removeItem('user');
    if (this.refreshTimeout) {
      clearTimeout(this.refreshTimeout);
      this.refreshTimeout = null;
    }
  }

  getTenantCode() {
    return localStorage.getItem(TENANT_CODE_KEY);
  }

  decodeToken(token) {
    try {
      return jwtDecode(token);
    } catch {
      return null;
    }
  }

  getTokenExpiry(token) {
    const decoded = this.decodeToken(token);
    return decoded ? decoded.exp * 1000 : 0;
  }

  isTokenValid(token = null) {
    const tokenToCheck = token || this.getToken();
    if (!tokenToCheck) return false;
    
    const expiry = this.getTokenExpiry(tokenToCheck);
    return expiry > Date.now();
  }

  scheduleRefresh() {
    const token = this.getToken();
    if (!token) return;
    
    const expiry = this.getTokenExpiry(token);
    const now = Date.now();
    const refreshIn = expiry - now - this.refreshThreshold;
    
    if (this.refreshTimeout) clearTimeout(this.refreshTimeout);
    
    if (refreshIn > 0) {
      console.log(`Scheduling token refresh in ${Math.round(refreshIn / 1000 / 60)} minutes`);
      this.refreshTimeout = setTimeout(() => this.refreshToken(), refreshIn);
    } else {
      console.log('Token expires soon, refreshing immediately');
      this.refreshToken();
    }
  }

  async refreshToken() {
    if (this.isRefreshing) {
      // Wait for the current refresh to complete
      return new Promise((resolve) => {
        this.subscribe(resolve);
      });
    }

    this.isRefreshing = true;
    console.log('Refreshing token...');

    try {
      const token = this.getToken();
      if (!token) throw new Error('No token to refresh');

      const response = await fetch(`${this.getBackendUrl()}/api/refresh-token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          // REMOVED: X-Tenant-Code header - no longer needed!
        },
      });

      if (!response.ok) {
        throw new Error(`Refresh failed: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.access_token) {
        this.setToken(data.access_token);
        this.notifySubscribers(true);
        console.log('Token refreshed successfully');
        return data.access_token;
      } else {
        throw new Error('No access_token in refresh response');
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
      this.clearToken();
      this.notifySubscribers(false);
      window.location.href = '/login';
      throw error;
    } finally {
      this.isRefreshing = false;
    }
  }

  subscribe(callback) {
    this.subscribers.push(callback);
  }

  notifySubscribers(success) {
    this.subscribers.forEach(cb => cb(success));
    this.subscribers = [];
  }

  async fetchWithAuth(url, options = {}, retry = true) {
    let token = this.getToken();
    
    if (!token) {
      console.warn('No token available, redirecting to login');
      window.location.href = '/login';
      throw new Error('No authentication token');
    }

    // Check if token is about to expire or already expired
    if (!this.isTokenValid(token)) {
      console.log('Token expired or expiring soon, refreshing...');
      try {
        // If a refresh is already in progress, wait for it to complete
        if (this.isRefreshing) {
          console.log('Token refresh already in progress, waiting...');
          await new Promise((resolve) => {
            this.subscribe(resolve);
          });
          token = this.getToken();
        } else {
          await this.refreshToken();
          token = this.getToken();
        }
      } catch (error) {
        console.error('Failed to refresh expired token');
        window.location.href = '/login';
        throw error;
      }
    }

    // Prepare the full URL
    const fullUrl = url.startsWith('http') ? url : `${this.getBackendUrl()}${url}`;

    // Prepare headers - CRITICAL: Only Authorization header, NO X-Tenant-Code!
    // Don't set Content-Type if body is FormData (browser will set it with boundary)
    const headers = {
      'Accept': 'application/json',
      'Authorization': `Bearer ${token}`,
      // REMOVED: X-Tenant-Code header - backend gets tenant from JWT token now!
      ...options.headers, // Allow overriding default headers
    };

    // Only add Content-Type if body is not FormData
    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    const requestOptions = {
      ...options,
      headers,
    };

    try {
      console.log(`Making authenticated request to: ${fullUrl}`, {
        method: requestOptions.method,
        headers: {
          ...headers,
          Authorization: 'Bearer [REDACTED]' // Don't log the actual token
        }
      });
      
      const response = await fetch(fullUrl, requestOptions);
      
      console.log(`Response status for ${fullUrl}: ${response.status}`);

      // Handle 401 Unauthorized or 403 Forbidden
      if ((response.status === 401 || response.status === 403) && retry) {
        console.log(`Received ${response.status}, attempting token refresh...`);
        try {
          // If a refresh is already in progress, wait for it to complete
          if (this.isRefreshing) {
            console.log('Token refresh already in progress, waiting...');
            await new Promise((resolve) => {
              this.subscribe(resolve);
            });
          } else {
            await this.refreshToken();
          }
          // Retry the request with the new token
          return this.fetchWithAuth(url, options, false);
        } catch (refreshError) {
          console.error(`Token refresh failed after ${response.status}:`, refreshError);
          // Clear token and redirect to login
          this.clearToken();
          window.location.href = '/login';
          throw refreshError;
        }
      }

      // If we get a 403 on retry, clear token and redirect
      if (response.status === 403 && !retry) {
        console.error('Still getting 403 after token refresh, clearing token');
        this.clearToken();
        window.location.href = '/login';
        throw new Error('Authorization failed');
      }

      return response;
    } catch (error) {
      console.error('Authenticated request failed:', error);
      // If it's a network error or other non-HTTP error, throw it
      throw error;
    }
  }

  // Convenience methods for common HTTP operations
  async get(url, options = {}) {
    return this.fetchWithAuth(url, { ...options, method: 'GET' });
  }

  async post(url, data, options = {}) {
    const requestOptions = {
      ...options,
      method: 'POST',
      // If data is FormData, don't stringify it
      body: data instanceof FormData ? data : (data ? JSON.stringify(data) : undefined)
    };

    return this.fetchWithAuth(url, requestOptions);
  }

  async put(url, data, options = {}) {
    const requestOptions = {
      ...options,
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined
    };

    return this.fetchWithAuth(url, requestOptions);
  }

  async delete(url, options = {}) {
    return this.fetchWithAuth(url, { ...options, method: 'DELETE' });
  }

  // Helper method to check if user is logged in
  isLoggedIn() {
    const token = this.getToken();
    return token && this.isTokenValid(token);
  }

  // Helper method to get user info from token
  getUserInfo() {
    const token = this.getToken();
    if (!token) return null;
    
    const decoded = this.decodeToken(token);
    if (!decoded) return null;
    
    return {
      username: decoded.sub,
      tenant_code: decoded.tenant_code,
      role: decoded.role,
      exp: decoded.exp
    };
  }

  // Method to handle logout
  logout() {
    this.clearToken();
    window.location.href = '/login';
  }
}

const tokenManager = new TokenManager();
export default tokenManager;