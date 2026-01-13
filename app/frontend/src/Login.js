import { useNavigate, useLocation } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import './Login.css';
import tokenManager from './tokenManager';
import { useAuth } from './auth/AuthProvider';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();
    const location = useLocation();

    const {
        isAuthenticated,
        loginWithAzureAD,
        authConfig,
        isLoading: authLoading,
        error: authError
    } = useAuth();

    // Check for error passed from auth callback
    useEffect(() => {
        if (location.state?.error) {
            setError(location.state.error);
        }
    }, [location.state]);

    useEffect(() => {
        // Check if user is already logged in
        if (isAuthenticated || tokenManager.isLoggedIn()) {
            console.log('User already logged in, redirecting to dashboard');
            navigate('/');
        }
    }, [navigate, isAuthenticated]);

    useEffect(() => {
        if (authError) {
            setError(authError);
        }
    }, [authError]);

    const handleLocalSubmit = async (event) => {
        event.preventDefault();

        if (!authConfig?.local_auth_enabled) {
            setError('Local authentication is disabled. Please use SSO.');
            return;
        }

        setError('');
        setIsLoading(true);

        try {
            const response = await fetch(`${tokenManager.getBackendUrl()}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                body: JSON.stringify({
                    username: username.trim(),
                    password: password
                }),
            });

            const data = await response.json();
            console.log('Login response:', data);

            if (response.ok && data.success) {
                console.log('Login successful, storing token and user data');

                tokenManager.setToken(data.access_token);

                localStorage.setItem('userName', username.trim());
                localStorage.setItem('tenantCode', data.tenant_code);
                localStorage.setItem('user', JSON.stringify({
                    username: data.username,
                    role: data.role,
                    tenant_code: data.tenant_code,
                    permissions: data.permissions,
                    authMethod: 'local',
                }));

                console.log('Authentication successful, navigating to dashboard');
                navigate('/');
            } else {
                console.error('Login failed:', data);
                setError(data.detail || 'Invalid credentials');
            }
        } catch (err) {
            console.error('Login error:', err);
            setError('Connection error. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleAzureADLogin = async () => {
        setError('');
        await loginWithAzureAD();
    };

    if (authLoading) {
        return (
            <div className="login-page">
                <div className="login-container">
                    <div className="loading-spinner"></div>
                    <p>Authenticating...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="login-page">
            <div className="login-container">
                <div className="company-logo">
                    <img src="/logo.png" alt="Company Logo" />
                </div>

                {/* Azure AD SSO Button */}
                {authConfig?.sso_enabled && (
                    <>
                        <button
                            className="azure-login-btn"
                            onClick={handleAzureADLogin}
                            disabled={isLoading}
                            type="button"
                        >
                            <svg className="microsoft-icon" viewBox="0 0 21 21" xmlns="http://www.w3.org/2000/svg">
                                <rect x="1" y="1" width="9" height="9" fill="#f25022"/>
                                <rect x="11" y="1" width="9" height="9" fill="#7fba00"/>
                                <rect x="1" y="11" width="9" height="9" fill="#00a4ef"/>
                                <rect x="11" y="11" width="9" height="9" fill="#ffb900"/>
                            </svg>
                            Sign in with Microsoft
                        </button>

                        {authConfig?.local_auth_enabled && (
                            <div className="login-divider">
                                <span>or</span>
                            </div>
                        )}
                    </>
                )}

                {/* Local Login Form */}
                {authConfig?.local_auth_enabled && (
                    <form onSubmit={handleLocalSubmit}>
                        <div className="input-group">
                            <label>Username</label>
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                                placeholder="Enter your username"
                                disabled={isLoading}
                            />
                        </div>
                        <div className="input-group">
                            <label>Password</label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                placeholder="Enter your password"
                                disabled={isLoading}
                            />
                        </div>
                        {error && <p className="error">{error}</p>}
                        <button type="submit" disabled={isLoading}>
                            {isLoading ? 'Logging in...' : 'Login'}
                        </button>
                    </form>
                )}

                {/* Error display for SSO-only mode */}
                {!authConfig?.local_auth_enabled && error && (
                    <p className="error">{error}</p>
                )}

                {!authConfig?.local_auth_enabled && !authConfig?.sso_enabled && (
                    <p className="error">No authentication methods available.</p>
                )}
            </div>
        </div>
    );
}

export default Login;
