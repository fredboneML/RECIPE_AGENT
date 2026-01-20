// auth/msalConfig.js
// MSAL configuration for Azure AD SSO integration
// NOTE: This file is dynamically imported only when SSO is enabled

/**
 * Create MSAL configuration from auth config fetched from backend
 * @param {Object} authConfig - Auth configuration from backend
 * @param {Object} LogLevel - MSAL LogLevel enum (passed from dynamic import)
 */
export const getMsalConfig = (authConfig, LogLevel) => ({
  auth: {
    clientId: authConfig.azure_ad.client_id,
    authority: authConfig.azure_ad.authority,
    redirectUri: authConfig.azure_ad.redirect_uri,
    postLogoutRedirectUri: window.location.origin,
    navigateToLoginRequestUrl: true,
  },
  cache: {
    cacheLocation: "sessionStorage", // More secure than localStorage
    storeAuthStateInCookie: false,
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) return;
        if (LogLevel) {
          switch (level) {
            case LogLevel.Error:
              console.error(message);
              break;
            case LogLevel.Warning:
              console.warn(message);
              break;
            case LogLevel.Info:
              console.info(message);
              break;
            default:
              console.debug(message);
          }
        } else {
          // Fallback if LogLevel not provided
          console.log(`[MSAL ${level}]`, message);
        }
      },
      logLevel: LogLevel?.Warning ?? 2, // 2 = Warning level
    },
  },
});

/**
 * Scopes for login request
 */
export const loginRequest = {
  scopes: ["openid", "profile", "email", "User.Read"],
};

/**
 * Scopes for token request
 */
export const tokenRequest = {
  scopes: ["openid", "profile", "email"],
};
