import React, { createContext, useState, useContext, ReactNode } from 'react';
import { translations } from './translations';

const LanguageContext = createContext({
  language: 'nl', // Default to Dutch
  setLanguage: () => {},
  t: (key) => key,
});

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState('nl'); // Default to Dutch

  const t = (key) => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};

export const useLanguage = () => useContext(LanguageContext); 