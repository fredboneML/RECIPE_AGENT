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

    // Render the comparison table with 60 specified fields
    const renderComparisonTable = () => {
        // DEBUG: Log BEFORE early return
        console.log('=== renderComparisonTable CALLED ===');
        console.log('comparisonTable:', comparisonTable);
        console.log('has_data:', comparisonTable?.has_data);
        
        if (!comparisonTable || !comparisonTable.has_data) {
            console.log('Early return: comparisonTable or has_data is falsy');
            return null;
        }

        // DEBUG: Log what we received
        console.log('=== COMPARISON TABLE DEBUG ===');
        console.log('comparisonTable:', comparisonTable);
        console.log('has field_definitions?', 'field_definitions' in comparisonTable);
        console.log('field_definitions:', comparisonTable.field_definitions);
        console.log('field_definitions length:', comparisonTable.field_definitions?.length);
        console.log('==============================');

        // Check if we have the new format with field_definitions
        const hasFieldDefinitions = comparisonTable.field_definitions && comparisonTable.field_definitions.length > 0;
        const recipes = comparisonTable.recipes || [];
        
        console.log('hasFieldDefinitions evaluated to:', hasFieldDefinitions);
        
        if (recipes.length === 0) return null;

        if (hasFieldDefinitions) {
            // New format: field_definitions + recipes with values array
            const fieldDefs = comparisonTable.field_definitions;
            
            return (
                <div className="comparison-table-container">
                    <h4>Recipe Comparison - 60 Specified Fields</h4>
                    <div className="table-wrapper">
                        <table className="comparison-table structured">
                            <thead>
                                <tr>
                                    <th className="field-code-header">Code</th>
                                    <th className="field-name-header">Field Name</th>
                                    {recipes.map((recipe, idx) => (
                                        <th key={idx} className="recipe-value-header">
                                            {recipe.recipe_name || recipe.recipe_id}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {fieldDefs.map((field, fieldIdx) => {
                                    // Check if any recipe has a value for this field
                                    const hasAnyValue = recipes.some(r => 
                                        r.values && r.values[fieldIdx] && r.values[fieldIdx].trim() !== ''
                                    );
                                    
                                    return (
                                        <tr key={fieldIdx} className={hasAnyValue ? 'has-value' : 'no-value'}>
                                            <td className="field-code">{field.code}</td>
                                            <td className="field-name">{field.display_name}</td>
                                            {recipes.map((recipe, recipeIdx) => (
                                                <td key={recipeIdx} className="recipe-value">
                                                    {recipe.values && recipe.values[fieldIdx] 
                                                        ? recipe.values[fieldIdx] 
                                                        : '-'}
                                                </td>
                                            ))}
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>
                    <div className="table-legend">
                        <span className="legend-item has-value-legend">● Fields with values</span>
                        <span className="legend-item no-value-legend">○ Fields without values</span>
                    </div>
                </div>
            );
        } else {
            // Legacy format: recipes with characteristics array
            // Get all unique characteristics across all recipes
            const allChars = new Set();
            recipes.forEach(recipe => {
                (recipe.characteristics || []).forEach(char => {
                    if (char.charactDescr) allChars.add(char.charactDescr);
                });
            });
            const sortedChars = Array.from(allChars).sort();

            return (
                <div className="comparison-table-container">
                    <h4>Recipe Comparison</h4>
                    <div className="table-wrapper">
                        <table className="comparison-table legacy">
                            <thead>
                                <tr>
                                    <th className="field-name-header">Characteristic</th>
                                    {recipes.map((recipe, idx) => (
                                        <th key={idx} className="recipe-value-header">
                                            {recipe.recipe_name || recipe.recipe_id}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {sortedChars.map((charName, charIdx) => (
                                    <tr key={charIdx}>
                                        <td className="field-name">{charName}</td>
                                        {recipes.map((recipe, recipeIdx) => {
                                            const char = (recipe.characteristics || []).find(
                                                c => c.charactDescr === charName
                                            );
                                            return (
                                                <td key={recipeIdx} className="recipe-value">
                                                    {char?.valueCharLong || '-'}
                                                </td>
                                            );
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            );
        }
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

            {renderComparisonTable()}

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
