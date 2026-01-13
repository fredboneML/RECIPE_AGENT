// auth/AuthCallback.js
// Handles the redirect callback from Azure AD authentication

import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from './AuthProvider';
import './AuthCallback.css';

function AuthCallback() {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading, error } = useAuth();

  useEffect(() => {
    // Wait for auth to complete processing
    if (!isLoading) {
      if (isAuthenticated) {
        // Successfully authenticated, redirect to main app
        navigate('/', { replace: true });
      } else if (error) {
        // Authentication failed, redirect to login with error
        console.error('Authentication callback error:', error);
        navigate('/login', { replace: true, state: { error } });
      }
      // If neither authenticated nor error, we're still processing
      // The AuthProvider will update state when done
    }
  }, [isAuthenticated, isLoading, error, navigate]);

  return (
    <div className="auth-callback">
      <div className="callback-container">
        <div className="spinner"></div>
        <p>Completing authentication...</p>
        {error && <p className="error-message">{error}</p>}
      </div>
    </div>
  );
}

export default AuthCallback;
