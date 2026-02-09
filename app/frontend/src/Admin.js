import { useNavigate } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import './Admin.css';
import tokenManager from './tokenManager';

function Admin() {
  const [users, setUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [showAddUser, setShowAddUser] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState('read_only');
  const [autoGeneratePassword, setAutoGeneratePassword] = useState(true);
  const [resetPasswordUsername, setResetPasswordUsername] = useState('');
  const [resetPassword, setResetPassword] = useState('');
  const [autoGenerateResetPassword, setAutoGenerateResetPassword] = useState(true);
  const [showResetPassword, setShowResetPassword] = useState(false);
  const [generatedPassword, setGeneratedPassword] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const verifyAuth = async () => {
      try {
        const userStr = localStorage.getItem('user');
        if (!userStr) {
          navigate('/login', { replace: true });
          return;
        }

        const user = JSON.parse(userStr);
        if (user.role !== 'admin') {
          setError('Access denied. Admin privileges required.');
          setTimeout(() => {
            navigate('/', { replace: true });
          }, 2000);
          return;
        }

        await loadUsers();
      } catch (error) {
        console.error('Auth verification failed:', error);
        navigate('/login', { replace: true });
      } finally {
        setIsLoading(false);
      }
    };

    verifyAuth();
  }, [navigate]);

  const loadUsers = async () => {
    try {
      setError('');
      const response = await tokenManager.get('/admin/users');
      if (!response.ok) {
        if (response.status === 403) {
          setError('Access denied. Admin privileges required.');
          return;
        }
        throw new Error('Failed to load users');
      }
      const data = await response.json();
      setUsers(data.users || []);
    } catch (err) {
      console.error('Error loading users:', err);
      setError('Failed to load users. Please try again.');
    }
  };

  const handleAddUser = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    try {
      const response = await tokenManager.post('/admin/users', {
        username: newUsername.trim(),
        password: autoGeneratePassword ? null : newPassword,
        role: newRole
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to add user');
      }

      const data = await response.json();
      setSuccess(`User '${newUsername}' added successfully!`);
      
      if (data.generated_password) {
        setGeneratedPassword({
          username: newUsername,
          password: data.generated_password
        });
      }

      // Reset form
      setNewUsername('');
      setNewPassword('');
      setNewRole('read_only');
      setAutoGeneratePassword(true);
      setShowAddUser(false);

      // Reload users
      await loadUsers();
    } catch (err) {
      console.error('Error adding user:', err);
      setError(err.message || 'Failed to add user. Please try again.');
    }
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    try {
      const requestBody = autoGenerateResetPassword 
        ? {} 
        : { password: resetPassword };
      
      const response = await tokenManager.put(`/admin/users/${resetPasswordUsername}/reset-password`, requestBody);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to reset password');
      }

      const data = await response.json();
      setSuccess(`Password reset successfully for user '${resetPasswordUsername}'!`);
      
      if (data.generated_password) {
        setGeneratedPassword({
          username: resetPasswordUsername,
          password: data.generated_password
        });
      }

      // Reset form
      setResetPasswordUsername('');
      setResetPassword('');
      setAutoGenerateResetPassword(true);
      setShowResetPassword(false);

      // Reload users
      await loadUsers();
    } catch (err) {
      console.error('Error resetting password:', err);
      setError(err.message || 'Failed to reset password. Please try again.');
    }
  };

  const handleDeleteUser = async (username) => {
    if (!window.confirm(`Are you sure you want to delete user '${username}'? This action cannot be undone.`)) {
      return;
    }

    setError('');
    setSuccess('');

    try {
      const response = await tokenManager.delete(`/admin/users/${username}`);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete user');
      }

      setSuccess(`User '${username}' deleted successfully!`);
      await loadUsers();
    } catch (err) {
      console.error('Error deleting user:', err);
      setError(err.message || 'Failed to delete user. Please try again.');
    }
  };

  const handleLogout = () => {
    localStorage.clear();
    tokenManager.clearToken();
    navigate('/login');
  };

  const copyPassword = (password) => {
    navigator.clipboard.writeText(password);
    setSuccess('Password copied to clipboard!');
    setTimeout(() => {
      setGeneratedPassword(null);
      setSuccess('');
    }, 3000);
  };

  if (isLoading) {
    return (
      <div className="admin-page">
        <div className="admin-container">
          <div className="loading-spinner"></div>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="admin-page">
      <div className="admin-header">
        <div className="admin-header-left">
          <h1>User Management</h1>
        </div>
        <div className="admin-header-right">
          <button className="back-button" onClick={() => navigate('/')}>
            Back to App
          </button>
          <button className="logout-button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>

      <div className="admin-content">
        {error && (
          <div className="admin-message error-message">
            {error}
          </div>
        )}

        {success && !generatedPassword && (
          <div className="admin-message success-message">
            {success}
          </div>
        )}

        {generatedPassword && (
          <div className="admin-message password-message">
            <div className="password-display">
              <strong>Generated Password for '{generatedPassword.username}':</strong>
              <div className="password-box">
                <code>{generatedPassword.password}</code>
                <button onClick={() => copyPassword(generatedPassword.password)}>
                  Copy
                </button>
              </div>
              <p className="password-warning">⚠️ Save this password now. It will not be shown again.</p>
              <button onClick={() => setGeneratedPassword(null)}>Close</button>
            </div>
          </div>
        )}

        <div className="admin-actions">
          <button
            className="add-user-button"
            onClick={() => {
              setShowAddUser(true);
              setShowResetPassword(false);
            }}
          >
            + Add New User
          </button>
        </div>

        {showAddUser && (
          <div className="admin-form-container">
            <h2>Add New User</h2>
            <form onSubmit={handleAddUser} className="admin-form">
              <div className="form-group">
                <label>Username *</label>
                <input
                  type="text"
                  value={newUsername}
                  onChange={(e) => setNewUsername(e.target.value)}
                  required
                  placeholder="Enter username"
                />
              </div>

              <div className="form-group">
                <label>
                  <input
                    type="checkbox"
                    checked={autoGeneratePassword}
                    onChange={(e) => setAutoGeneratePassword(e.target.checked)}
                  />
                  Auto-generate password
                </label>
              </div>

              {!autoGeneratePassword && (
                <div className="form-group">
                  <label>Password *</label>
                  <input
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    required
                    placeholder="Enter password"
                  />
                </div>
              )}

              <div className="form-group">
                <label>Role *</label>
                <select
                  value={newRole}
                  onChange={(e) => setNewRole(e.target.value)}
                  required
                >
                  <option value="read_only">Read Only</option>
                  <option value="write">Write</option>
                  <option value="admin">Admin</option>
                </select>
              </div>

              <div className="form-actions">
                <button type="submit" className="submit-button">
                  Add User
                </button>
                <button
                  type="button"
                  className="cancel-button"
                  onClick={() => {
                    setShowAddUser(false);
                    setNewUsername('');
                    setNewPassword('');
                    setNewRole('read_only');
                    setAutoGeneratePassword(true);
                  }}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        {showResetPassword && (
          <div className="admin-form-container">
            <h2>Reset Password</h2>
            <form onSubmit={handleResetPassword} className="admin-form">
              <div className="form-group">
                <label>Username *</label>
                <input
                  type="text"
                  value={resetPasswordUsername}
                  onChange={(e) => setResetPasswordUsername(e.target.value)}
                  required
                  placeholder="Enter username"
                />
              </div>

              <div className="form-group">
                <label>
                  <input
                    type="checkbox"
                    checked={autoGenerateResetPassword}
                    onChange={(e) => setAutoGenerateResetPassword(e.target.checked)}
                  />
                  Auto-generate password
                </label>
              </div>

              {!autoGenerateResetPassword && (
                <div className="form-group">
                  <label>New Password *</label>
                  <input
                    type="password"
                    value={resetPassword}
                    onChange={(e) => setResetPassword(e.target.value)}
                    required
                    placeholder="Enter new password"
                  />
                </div>
              )}

              <div className="form-actions">
                <button type="submit" className="submit-button">
                  Reset Password
                </button>
                <button
                  type="button"
                  className="cancel-button"
                  onClick={() => {
                    setShowResetPassword(false);
                    setResetPasswordUsername('');
                    setResetPassword('');
                    setAutoGenerateResetPassword(true);
                  }}
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        <div className="users-table-container">
          <h2>Users ({users.length})</h2>
          {users.length === 0 ? (
            <p className="no-users">No users found.</p>
          ) : (
            <table className="users-table">
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Role</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => (
                  <tr key={user.id}>
                    <td>{user.username}</td>
                    <td>
                      <span className={`role-badge role-${user.role}`}>
                        {user.role}
                      </span>
                    </td>
                    <td>
                      <div className="action-buttons">
                        <button
                          className="reset-password-button"
                          onClick={() => {
                            setResetPasswordUsername(user.username);
                            setShowResetPassword(true);
                            setShowAddUser(false);
                          }}
                        >
                          Reset Password
                        </button>
                        <button
                          className="delete-button"
                          onClick={() => handleDeleteUser(user.username)}
                          disabled={user.username === JSON.parse(localStorage.getItem('user') || '{}').username}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

export default Admin;
