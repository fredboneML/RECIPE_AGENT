// auth/msalConfig.js
// MSAL configuration for Azure AD SSO integration

import { LogLevel } from "@azure/msal-browser";

/**
 * Create MSAL configuration from auth config fetched from backend
 */
export const getMsalConfig = (authConfig) => ({
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
      },
      logLevel: LogLevel.Warning,
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
