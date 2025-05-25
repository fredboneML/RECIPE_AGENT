import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import './ChatInterface.css';

function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [userInput, setUserInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [conversationId, setConversationId] = useState(null);
    const [followupQuestions, setFollowupQuestions] = useState([]);
    const [contextExceeded, setContextExceeded] = useState(false);
    const isDutch = localStorage.getItem('language') === 'nl';

    const handleSendMessage = async (message) => {
        if (!message.trim()) return;

        setIsLoading(true);
        setError(null);

        try {
            const response = await api.query(message, conversationId);
            
            setMessages(prev => [...prev, 
                { type: 'user', content: message },
                { type: 'assistant', content: response.response }
            ]);

            if (response.conversation_id) {
                setConversationId(response.conversation_id);
            }

            if (response.followup_questions) {
                setFollowupQuestions(response.followup_questions);
            }

            setUserInput('');
        } catch (err) {
            console.error('Error sending message:', err);
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleNewConversation = () => {
        setMessages([]);
        setConversationId(null);
        setFollowupQuestions([]);
        setContextExceeded(false);
    };

    const handleFollowupClick = (question) => {
        if (contextExceeded && 
            (question.toLowerCase().includes("start a new conversation") || 
             question.toLowerCase().includes("start een nieuw gesprek"))) {
            handleNewConversation();
            return;
        }
        
        setUserInput(question);
        handleSendMessage(question);
    };

    return (
        <div className="chat-interface">
            <div className="messages-container">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.type}`}>
                        {msg.content}
                    </div>
                ))}
                {isLoading && <div className="loading">...</div>}
                {error && <div className="error">{error}</div>}
            </div>

            {followupQuestions && followupQuestions.length > 0 && (
                <div className="followup-questions">
                    <h4>{contextExceeded ? 
                        (isDutch ? "Aanbevolen acties:" : "Recommended actions:") : 
                        (isDutch ? "Suggesties voor vervolgvragen:" : "Suggested follow-up questions:")}
                    </h4>
                    <ul>
                        {followupQuestions.map((question, index) => (
                            <li 
                                key={index} 
                                onClick={() => handleFollowupClick(question)}
                                className={contextExceeded && 
                                    (question.toLowerCase().includes("start a new conversation") || 
                                     question.toLowerCase().includes("start een nieuw gesprek")) 
                                    ? "new-conversation-action" : ""}
                            >
                                {question}
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <div className="input-container">
                <input
                    type="text"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(userInput)}
                    placeholder={isDutch ? "Stel een vraag..." : "Ask a question..."}
                    disabled={isLoading}
                />
                <button 
                    onClick={() => handleSendMessage(userInput)}
                    disabled={isLoading || !userInput.trim()}
                >
                    {isLoading ? '...' : 'Send'}
                </button>
                <button 
                    onClick={handleNewConversation}
                    className="new-conversation"
                >
                    {isDutch ? "Nieuw gesprek" : "New Conversation"}
                </button>
            </div>
        </div>
    );
}

export default ChatInterface; 