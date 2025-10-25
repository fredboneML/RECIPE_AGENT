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
    const [comparisonTable, setComparisonTable] = useState(null);

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

            if (response.comparison_table) {
                console.log('Comparison table received:', response.comparison_table);
                setComparisonTable(response.comparison_table);
            } else {
                console.log('No comparison table in response');
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
        setComparisonTable(null);
    };

    const handleFollowupClick = (question) => {
        if (contextExceeded && question.toLowerCase().includes("start a new conversation")) {
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

            {comparisonTable && comparisonTable.has_data && (
                <div className="comparison-table-container">
                    <h4>Recipe Comparison</h4>
                    <div className="table-wrapper">
                        <table className="comparison-table">
                            <thead>
                                <tr className="recipe-names">
                                    {comparisonTable.recipes.map((recipe, index) => (
                                        <th key={index} colSpan="2" className="recipe-header">
                                            {recipe.recipe_name || recipe.recipe_id}
                                        </th>
                                    ))}
                                </tr>
                                <tr className="column-headers">
                                    {comparisonTable.recipes.map((recipe, index) => (
                                        <React.Fragment key={index}>
                                            <th className="characteristic-header">Characteristic</th>
                                            <th className="value-header">Value</th>
                                        </React.Fragment>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {comparisonTable.recipes[0]?.characteristics.map((_, charIndex) => (
                                    <tr key={charIndex}>
                                        {comparisonTable.recipes.map((recipe, recipeIndex) => (
                                            <React.Fragment key={recipeIndex}>
                                                <td className="characteristic-cell">
                                                    {recipe.characteristics[charIndex]?.charactDescr || ''}
                                                </td>
                                                <td className="value-cell">
                                                    {recipe.characteristics[charIndex]?.valueCharLong || ''}
                                                </td>
                                            </React.Fragment>
                                        ))}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {followupQuestions && followupQuestions.length > 0 && (
                <div className="followup-questions">
                    <h4>{contextExceeded ? "Recommended actions:" : "Suggested follow-up questions:"}</h4>
                    <ul>
                        {followupQuestions.map((question, index) => (
                            <li 
                                key={index} 
                                onClick={() => handleFollowupClick(question)}
                                className={contextExceeded && question.toLowerCase().includes("start a new conversation") ? "new-conversation-action" : ""}
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
                    placeholder="Ask a question..."
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
                    New Conversation
                </button>
            </div>
        </div>
    );
}

export default ChatInterface; 