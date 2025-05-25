import { jwtDecode } from 'jwt-decode';

const TOKEN_KEY = 'token';
const TENANT_CODE_KEY = 'tenantCode';
const REFRESH_ENDPOINT = '/api/refresh-token';

class TokenManager {
  constructor() {
    this.refreshTimeout = null;
    this.refreshThreshold = 5 * 60 * 1000; // 5 minutes before expiry
    this.isRefreshing = false;
    this.subscribers = [];
  }

  getToken() {
    return localStorage.getItem(TOKEN_KEY);
  }

  setToken(token) {
    localStorage.setItem(TOKEN_KEY, token);
  }

  clearToken() {
    localStorage.removeItem(TOKEN_KEY);
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

  scheduleRefresh() {
    const token = this.getToken();
    if (!token) return;
    const expiry = this.getTokenExpiry(token);
    const now = Date.now();
    const refreshIn = expiry - now - this.refreshThreshold;
    if (this.refreshTimeout) clearTimeout(this.refreshTimeout);
    if (refreshIn > 0) {
      this.refreshTimeout = setTimeout(() => this.refreshToken(), refreshIn);
    }
  }

  async refreshToken() {
    if (this.isRefreshing) return;
    this.isRefreshing = true;
    try {
      const token = this.getToken();
      if (!token) throw new Error('No token');
      const response = await fetch(REFRESH_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
          'X-Tenant-Code': this.getTenantCode(),
        },
      });
      if (!response.ok) throw new Error('Refresh failed');
      const data = await response.json();
      if (data.access_token) {
        this.setToken(data.access_token);
        this.scheduleRefresh();
        this.notifySubscribers(true);
      } else {
        throw new Error('No access_token in response');
      }
    } catch (e) {
      this.clearToken();
      this.notifySubscribers(false);
      window.location.href = '/login';
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
      window.location.href = '/login';
      throw new Error('No token');
    }
    // Check if token is about to expire
    const expiry = this.getTokenExpiry(token);
    if (expiry - Date.now() < this.refreshThreshold) {
      await this.refreshToken();
      token = this.getToken();
    }
    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${token}`,
      'X-Tenant-Code': this.getTenantCode(),
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    };
    try {
      const response = await fetch(url, { ...options, headers });
      if (response.status === 401 && retry) {
        await this.refreshToken();
        return this.fetchWithAuth(url, options, false);
      }
      return response;
    } catch (e) {
      throw e;
    }
  }
}

const tokenManager = new TokenManager();
export default tokenManager;