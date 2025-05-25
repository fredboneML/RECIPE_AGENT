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

    // Dynamically determine the backend URL
    const backendUrl = window.location.hostname === 'localhost' 
        ? 'http://localhost:8000'
        : `http://${window.location.hostname}:8000`;

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            navigate('/');
        }
    }, [navigate]);

    const handleSubmit = async (event) => {
        event.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            console.log('Attempting to connect to:', backendUrl);
            const response = await fetch(`${backendUrl}/api/login`, {
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

            if (response.ok && data.success) {
                // Store the access token
                tokenManager.setToken(data.access_token);
                localStorage.setItem('userName', username.trim());
                localStorage.setItem('tenantCode', data.tenant_code);
                localStorage.setItem('user', JSON.stringify({
                    username: data.username,
                    role: data.role,
                    tenant_code: data.tenant_code,
                    permissions: data.permissions
                }));
                tokenManager.scheduleRefresh();
                navigate('/');
            } else {
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