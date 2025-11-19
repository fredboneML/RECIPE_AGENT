import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import './Login.css';
import tokenManager from './tokenManager';

function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        // Check if user is already logged in
        if (tokenManager.isLoggedIn()) {
            console.log('User already logged in, redirecting to dashboard');
            navigate('/');
        }
    }, [navigate]);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            // Login endpoint doesn't need authentication, so we use a special method
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
                
                // Store the access token using tokenManager
                tokenManager.setToken(data.access_token);
                
                // Store additional user data in localStorage
                localStorage.setItem('userName', username.trim());
                localStorage.setItem('tenantCode', data.tenant_code);
                localStorage.setItem('user', JSON.stringify({
                    username: data.username,
                    role: data.role,
                    tenant_code: data.tenant_code,
                    permissions: data.permissions
                }));
                
                // Token stored successfully, navigate to dashboard
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

    return (
        <div className="login-page">
            <div className="login-container">
                <div className="company-logo">
                    <img src="/logo.png" alt="Company Logo" />
                </div>
                <form onSubmit={handleSubmit}>
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
            </div>
        </div>
    );
}

export default Login;