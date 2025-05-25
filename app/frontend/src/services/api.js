// API service for making authenticated requests
const backendUrl = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'
    : `http://${window.location.hostname}:8000`;

const getHeaders = () => {
    const token = localStorage.getItem('token');
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': token ? `Bearer ${token}` : '',
    };
};

export const api = {
    async query(question, conversationId = null) {
        const response = await fetch(`${backendUrl}/api/query`, {
            method: 'POST',
            headers: getHeaders(),
            body: JSON.stringify({
                query: question,
                conversation_id: conversationId
            }),
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                // Token expired or invalid
                localStorage.removeItem('token');
                window.location.href = '/login';
                throw new Error('Session expired. Please login again.');
            }
            throw new Error('Failed to get response');
        }
        
        return response.json();
    },

    async getConversations() {
        const response = await fetch(`${backendUrl}/api/conversations`, {
            headers: getHeaders(),
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('token');
                window.location.href = '/login';
                throw new Error('Session expired. Please login again.');
            }
            throw new Error('Failed to get conversations');
        }
        
        return response.json();
    },

    async getConversation(conversationId) {
        const response = await fetch(`${backendUrl}/api/conversations/${conversationId}`, {
            headers: getHeaders(),
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                localStorage.removeItem('token');
                window.location.href = '/login';
                throw new Error('Session expired. Please login again.');
            }
            throw new Error('Failed to get conversation');
        }
        
        return response.json();
    }
}; 