// auth/AuthProvider.js
// React context provider for Azure AD SSO authentication

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { PublicClientApplication, InteractionStatus } from "@azure/msal-browser";
import { MsalProvider, useMsal, useIsAuthenticated } from "@azure/msal-react";
import { getMsalConfig, loginRequest } from './msalConfig';
import tokenManager from '../tokenManager';

const AuthContext = createContext(null);

/**
 * Inner provider that uses MSAL hooks (must be inside MsalProvider)
 */
function AuthProviderInner({ children, authConfig }) {
  const { instance, accounts, inProgress } = useMsal();
  const isAzureAuthenticated = useIsAuthenticated();
  const [authState, setAuthState] = useState({
    isLoading: true,
    isAuthenticated: false,
    user: null,
    error: null,
    authMethod: null, // 'azure_ad' or 'local'
  });

  // Handle Azure AD login with redirect
  const loginWithAzureAD = useCallback(async () => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true, error: null }));
      await instance.loginRedirect(loginRequest);
    } catch (error) {
      console.error('Azure AD login error:', error);
      setAuthState(prev => ({
        ...prev,
        isLoading: false,
        error: error.message,
      }));
    }
  }, [instance]);

  // Handle redirect response from Azure AD
  useEffect(() => {
    const handleRedirectResponse = async () => {
      if (inProgress !== InteractionStatus.None) return;

      try {
        // Check for redirect response
        const response = await instance.handleRedirectPromise();

        if (response) {
          // Successfully authenticated with Azure AD
          console.log('Azure AD authentication successful');

          // Send ID token to backend for validation
          const backendResponse = await fetch(
            `${tokenManager.getBackendUrl()}/auth/azure-callback`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ id_token: response.idToken }),
            }
          );

          if (!backendResponse.ok) {
            const errorData = await backendResponse.json();
            throw new Error(errorData.detail || 'Backend authentication failed');
          }

          const data = await backendResponse.json();

          // Store internal JWT
          tokenManager.setToken(data.access_token);
          localStorage.setItem('userName', data.username);
          localStorage.setItem('user', JSON.stringify({
            username: data.username,
            email: data.email,
            role: data.role,
            permissions: data.permissions,
            authMethod: 'azure_ad',
          }));

          setAuthState({
            isLoading: false,
            isAuthenticated: true,
            user: data,
            error: null,
            authMethod: 'azure_ad',
          });
        } else if (accounts.length > 0 && isAzureAuthenticated) {
          // Already authenticated with Azure AD - try to get token silently
          try {
            const silentResponse = await instance.acquireTokenSilent({
              ...loginRequest,
              account: accounts[0],
            });

            // Validate with backend
            const backendResponse = await fetch(
              `${tokenManager.getBackendUrl()}/auth/azure-callback`,
              {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id_token: silentResponse.idToken }),
              }
            );

            if (backendResponse.ok) {
              const data = await backendResponse.json();
              tokenManager.setToken(data.access_token);
              localStorage.setItem('userName', data.username);
              localStorage.setItem('user', JSON.stringify({
                username: data.username,
                email: data.email,
                role: data.role,
                permissions: data.permissions,
                authMethod: 'azure_ad',
              }));

              setAuthState({
                isLoading: false,
                isAuthenticated: true,
                user: data,
                error: null,
                authMethod: 'azure_ad',
              });
              return;
            }
          } catch (silentError) {
            console.log('Silent token acquisition failed:', silentError.message);
          }
        }

        // Check for existing local auth
        if (tokenManager.isLoggedIn()) {
          const userStr = localStorage.getItem('user');
          if (userStr) {
            try {
              const user = JSON.parse(userStr);
              setAuthState({
                isLoading: false,
                isAuthenticated: true,
                user,
                error: null,
                authMethod: user.authMethod || 'local',
              });
              return;
            } catch (e) {
              console.error('Error parsing user data:', e);
            }
          }
        }

        // Not authenticated
        setAuthState(prev => ({ ...prev, isLoading: false }));

      } catch (error) {
        console.error('Auth error:', error);
        setAuthState({
          isLoading: false,
          isAuthenticated: false,
          user: null,
          error: error.message,
          authMethod: null,
        });
      }
    };

    handleRedirectResponse();
  }, [instance, accounts, inProgress, isAzureAuthenticated]);

  // Logout function
  const logout = useCallback(async () => {
    tokenManager.clearToken();
    localStorage.clear();

    if (authState.authMethod === 'azure_ad') {
      // Logout from Azure AD
      try {
        await instance.logoutRedirect({
          postLogoutRedirectUri: window.location.origin + '/login',
        });
      } catch (error) {
        console.error('Azure AD logout error:', error);
        window.location.href = '/login';
      }
    } else {
      window.location.href = '/login';
    }
  }, [instance, authState.authMethod]);

  // Check if logged in (for tokenManager compatibility)
  const isLoggedIn = useCallback(() => {
    return authState.isAuthenticated || tokenManager.isLoggedIn();
  }, [authState.isAuthenticated]);

  return (
    <AuthContext.Provider
      value={{
        ...authState,
        loginWithAzureAD,
        logout,
        isLoggedIn,
        authConfig,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

/**
 * Simple provider for when SSO is disabled (no MSAL)
 */
function SimpleAuthProvider({ children, authConfig }) {
  const [authState, setAuthState] = useState({
    isLoading: false,
    isAuthenticated: tokenManager.isLoggedIn(),
    user: null,
    error: null,
    authMethod: 'local',
  });

  useEffect(() => {
    if (tokenManager.isLoggedIn()) {
      const userStr = localStorage.getItem('user');
      if (userStr) {
        try {
          setAuthState(prev => ({
            ...prev,
            isAuthenticated: true,
            user: JSON.parse(userStr),
          }));
        } catch (e) {
          console.error('Error parsing user data:', e);
        }
      }
    }
  }, []);

  const logout = useCallback(() => {
    tokenManager.clearToken();
    localStorage.clear();
    window.location.href = '/login';
  }, []);

  return (
    <AuthContext.Provider
      value={{
        ...authState,
        loginWithAzureAD: () => console.warn('SSO not enabled'),
        logout,
        isLoggedIn: () => tokenManager.isLoggedIn(),
        authConfig,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

/**
 * Main AuthProvider component
 * Fetches auth config from backend and initializes MSAL if SSO is enabled
 */
export function AuthProvider({ children }) {
  const [msalInstance, setMsalInstance] = useState(null);
  const [authConfig, setAuthConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [initError, setInitError] = useState(null);

  useEffect(() => {
    const initAuth = async () => {
      try {
        // Fetch auth configuration from backend
        const response = await fetch(`${tokenManager.getBackendUrl()}/auth/config`);

        if (!response.ok) {
          throw new Error('Failed to fetch auth configuration');
        }

        const config = await response.json();
        setAuthConfig(config);

        if (config.sso_enabled && config.azure_ad) {
          // Initialize MSAL
          const msalConfig = getMsalConfig(config);
          const pca = new PublicClientApplication(msalConfig);
          await pca.initialize();
          setMsalInstance(pca);
          console.log('MSAL initialized successfully');
        }
      } catch (error) {
        console.error('Failed to initialize auth:', error);
        setInitError(error.message);
        // Set default config to allow local auth
        setAuthConfig({
          sso_enabled: false,
          local_auth_enabled: true,
          azure_ad: null,
        });
      } finally {
        setLoading(false);
      }
    };

    initAuth();
  }, []);

  if (loading) {
    return (
      <div className="auth-loading">
        <div className="loading-spinner"></div>
        <p>Initializing...</p>
      </div>
    );
  }

  if (initError && !authConfig?.local_auth_enabled) {
    return (
      <div className="auth-error">
        <p>Authentication initialization failed: {initError}</p>
      </div>
    );
  }

  // Use MSAL provider if SSO is enabled
  if (msalInstance) {
    return (
      <MsalProvider instance={msalInstance}>
        <AuthProviderInner authConfig={authConfig}>
          {children}
        </AuthProviderInner>
      </MsalProvider>
    );
  }

  // Use simple provider without MSAL
  return (
    <SimpleAuthProvider authConfig={authConfig}>
      {children}
    </SimpleAuthProvider>
  );
}

/**
 * Hook to access auth context
 */
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};

export default AuthProvider;
