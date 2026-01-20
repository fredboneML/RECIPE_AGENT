// auth/AuthProvider.js
// React context provider for authentication with optional Azure AD SSO

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import tokenManager from '../tokenManager';

const AuthContext = createContext(null);

/**
 * Simple provider for local auth (no MSAL dependencies)
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
        loginWithAzureAD: () => console.warn('SSO not enabled - use username/password'),
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
 * MSAL-enabled provider (loaded dynamically)
 */
function createMsalAuthProvider(msalBrowser, msalReact, getMsalConfig, loginRequest) {
  const { PublicClientApplication, InteractionStatus, LogLevel } = msalBrowser;
  const { MsalProvider, useMsal, useIsAuthenticated } = msalReact;
  
  // Create config factory that includes LogLevel
  const msalConfig = (authConfig) => getMsalConfig(authConfig, LogLevel);

  function AuthProviderInner({ children, authConfig }) {
    const { instance, accounts, inProgress } = useMsal();
    const isAzureAuthenticated = useIsAuthenticated();
    const [authState, setAuthState] = useState({
      isLoading: true,
      isAuthenticated: false,
      user: null,
      error: null,
      authMethod: null,
    });

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

    useEffect(() => {
      const handleRedirectResponse = async () => {
        if (inProgress !== InteractionStatus.None) return;

        try {
          const response = await instance.handleRedirectPromise();

          if (response) {
            console.log('Azure AD authentication successful');
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
            try {
              const silentResponse = await instance.acquireTokenSilent({
                ...loginRequest,
                account: accounts[0],
              });

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

    const logout = useCallback(async () => {
      tokenManager.clearToken();
      localStorage.clear();

      if (authState.authMethod === 'azure_ad') {
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

  return function MsalAuthProvider({ children, authConfig, config }) {
    const [msalInstance, setMsalInstance] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
      const initMsal = async () => {
        try {
          const pca = new PublicClientApplication(msalConfig(config));
          await pca.initialize();
          setMsalInstance(pca);
          console.log('MSAL initialized successfully');
        } catch (err) {
          console.error('MSAL initialization failed:', err);
          setError(err.message);
        }
      };
      initMsal();
    }, [config]);

    if (error) {
      // Fallback to simple auth if MSAL fails
      return (
        <SimpleAuthProvider authConfig={authConfig}>
          {children}
        </SimpleAuthProvider>
      );
    }

    if (!msalInstance) {
      return (
        <div className="auth-loading">
          <div className="loading-spinner"></div>
          <p>Initializing SSO...</p>
        </div>
      );
    }

    return (
      <MsalProvider instance={msalInstance}>
        <AuthProviderInner authConfig={authConfig}>
          {children}
        </AuthProviderInner>
      </MsalProvider>
    );
  };
}

/**
 * Main AuthProvider component
 * Fetches auth config from backend and dynamically loads MSAL if SSO is enabled
 */
export function AuthProvider({ children }) {
  const [authConfig, setAuthConfig] = useState(null);
  const [MsalAuthProvider, setMsalAuthProvider] = useState(null);
  const [loading, setLoading] = useState(true);
  const [msalAvailable, setMsalAvailable] = useState(false);

  useEffect(() => {
    const initAuth = async () => {
      try {
        // Fetch auth configuration from backend
        const response = await fetch(`${tokenManager.getBackendUrl()}/auth/config`);

        if (!response.ok) {
          console.warn('Auth config endpoint not available, using local auth');
          setAuthConfig({
            sso_enabled: false,
            local_auth_enabled: true,
            azure_ad: null,
          });
          setLoading(false);
          return;
        }

        const config = await response.json();
        setAuthConfig(config);

        // Only try to load MSAL if SSO is enabled
        if (config.sso_enabled && config.azure_ad) {
          try {
            // Dynamically import MSAL packages
            const [msalBrowser, msalReact, msalConfigModule] = await Promise.all([
              import('@azure/msal-browser'),
              import('@azure/msal-react'),
              import('./msalConfig'),
            ]);

            const Provider = createMsalAuthProvider(
              msalBrowser,
              msalReact,
              msalConfigModule.getMsalConfig,
              msalConfigModule.loginRequest
            );
            setMsalAuthProvider(() => Provider);
            setMsalAvailable(true);
            console.log('MSAL packages loaded successfully');
          } catch (msalError) {
            console.warn('MSAL packages not available, using local auth:', msalError.message);
            // SSO requested but MSAL not available - fall back to local
          }
        }
      } catch (error) {
        console.error('Failed to initialize auth:', error);
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

  // Use MSAL provider if available and SSO is enabled
  if (msalAvailable && MsalAuthProvider && authConfig?.sso_enabled) {
    return (
      <MsalAuthProvider authConfig={authConfig} config={authConfig}>
        {children}
      </MsalAuthProvider>
    );
  }

  // Use simple provider without MSAL (local auth)
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
